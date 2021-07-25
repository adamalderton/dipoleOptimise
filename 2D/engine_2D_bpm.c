/****************************************************************************************
 * OpDDA_engine -- Optimises a set of discrete dipoles for a        *
 *                 dedicated purpose.                               *
 *                                                                  *
 * Author:                                                          *
 *          Adam Alderton (aa816@exeter.ac.uk)                      *
 *                                                                  *
 * Revision Date:                                                   *
 *          09/20/2020                                              *
 *                                                                  *
 * Usage:                                                           *
 *          This is a core "engine" which should be leveraged by    *
 *          the "OpDDA.py" python controller program. This core     *
 *          program should also facilitate the placement of dipoles *
 *          along with any results analysis. The outer controller   *
 *          should also specify any required run-time parameters    *
 *          and flags, such as whether to store all data or only    *
 *          required data.                                          *
 *                                                                  *
 ****************************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <json-c/json.h>

#include <gsl/gsl_math.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_matrix_complex_double.h>
#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_fft_complex.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>

#define FILENAME_BUFFER 128
#define LINE_BUFFER_LENGTH 1024

#define DIMS 2
#define POINTS_PER_DIM 3

#define POINTS_AROUND 9 /* 3^2 */

#define X 0
#define Z 1

/* 
 * Every function returns an errorcode. This macro is called after every function return
 * and checks if there was an error. If so, this ends the function which called the error returning function
 * and returns the error code.
 * Therefore, this macro passes the error code up the stack until it is eventually returned by main.
 */
#define ERROR_CHECK(error) \
    if (error != NO_ERROR) { \
        return error; \
    }

/* Checks for the success of the opening of a file. */
#define FILE_CHECK(file, filename) \
    if (file == NULL) { \
        fprintf(stderr, "Error: Unable to open file \"%s\".\n", filename);\
        return BAD_FILENAME; \
    }

/* Checks the success of a line read from a file. */
#define READLINE_CHECK(result, in_file) \
    if (result == NULL) { \
        fprintf(stderr, "Error: Could not read file.\n"); \
        fclose(in_file); \
        return BAD_FILE; \
    }

/* Checks the success of a strtok() call. */
#define STRTOK_CHECK(string, filename, in_file) \
    if (string == NULL) { \
        fprintf(stderr, "Error: Data could not be read from file \"%s\".\n", filename); \
        fclose(in_file); \
        return BAD_FILE; \
    }

#define MKDIR_CHECK(dir_name, result) \
    if (result != NO_ERROR) { \
        fprintf(stder, "Error: Could not create directory %s.\n", dir_name); \
        return BAD_DIR; \
    }

#define PACKED_REAL(z, i) ( (z)[2*(i)] )
#define PACKED_IMAG(z, i) ( (z)[2*(i) + 1] )

#define PACKED_SET_REAL(z, i, val) ((z)[2*(i)] = val)
#define PACKED_SET_IMAG(z, i, val) ((z)[2*(i) + 1] = val)

#define GET_KX(kx, max_x, x) \
    if (x < (max_x / 2)) { \
        *kx = (1.0 / max_x) * x; \
    } else { \
        *kx = ((1.0 / max_x) * x) - 1; \
    }

typedef enum {
    NO_ERROR = 0,
    BAD_FILE = 1025,
    BAD_FILENAME = 1026,
    BAD_ARGUMENTS = 1027,
    BAD_DIR = 1028
} ErrorCode;

/*
 * Structure to describe a scattering dipole.
 * Currently, both the polarisability and the polarisation are scalar.
 */
typedef struct {
    unsigned int _id;
    int position[DIMS];        /* 2D vector storing cartesian position. */
    int new_position[DIMS];
    float vdfs[POINTS_AROUND];  /* Virtual dipole fitnesses around dipoles */
    gsl_complex polarisation;
} Dipole;

typedef struct {
    unsigned int num_dipoles;
    double polarisability;
    Dipole *_dipoles;
} Dipoles;

/*
 * Structure to describe an input field.
 * Currently describes a plane wave (subject to change).
 */
typedef struct {
    int focus_point[DIMS];
    gsl_complex focus_point_field_input_only;
    gsl_complex_packed_array k_space;
} InputField;

typedef struct {
    char experiment_name[FILENAME_BUFFER];
    int detector_z;
    unsigned int num_inputs;
    unsigned int save_positions_every_X_epochs;
    unsigned int resolution_factor;
    unsigned int max_iterations;
    unsigned int scatterer_size[DIMS]; /* Dimensions of scatterer. */
    unsigned int extra_z_evaluation;
    double mod_k;                      /* Wavevector of all input fields. */
} Config;

/***********************************FUNCTION PROTOTYPES**********************************/
static ErrorCode parse_JSON(Dipoles *dipoles);
static ErrorCode unpack_dipoles(Dipoles *dipoles);
static ErrorCode unpack_input_fields(InputField **input_fields);
static ErrorCode save_positions(Dipoles dipoles, int epoch_num, double fitness, InputField input_field);
static ErrorCode process_results(Dipoles dipoles, InputField *input_fields);

static gsl_complex_packed_array evaluate_total_field_at_plane(Dipoles dipoles, InputField input_field, unsigned int z);
static gsl_complex_packed_array propagate_k_field_to_plane(InputField input_field, unsigned int z);
static gsl_complex propagate_k_field_to_k_point(InputField input_field, int point[DIMS]);
static gsl_complex evaluate_input_field_at_point(InputField input_field, int point[DIMS]);

static gsl_matrix_complex* find_A_inv(gsl_matrix_complex *A_inv, Dipoles dipoles);
static void find_polarisations(InputField input_field, Dipoles *dipoles, gsl_matrix_complex *A_inv);

static double evaluate_virtual_dipole(Dipole *virtual_dipole, Dipoles dipoles, InputField input_field, InputField *all_input_fields);
static void consider_dipole_with_input(Dipoles dipoles, Dipole *primary_dipole, InputField *all_input_fields, InputField input_field);
static void move_dipoles(Dipoles *dipoles);
static ErrorCode evaluate_progress(Dipoles *dipoles, InputField *input_fields, int epoch_num);
static ErrorCode optimise_dipoles(Dipoles *dipoles, InputField *input_fields);
/***************************************************************************************/

/* Global variables. */
Config config;

/*-----------------------------------IO FUNCTIONS---------------------------------------*/

static ErrorCode parse_JSON(Dipoles *dipoles)
{
    FILE *in_file;
    char filepath[FILENAME_BUFFER];
    char line_buffer[LINE_BUFFER_LENGTH];

    struct json_object *parsed_json;            /* Holds entire json file. */

    struct json_object *wavelength;             /* Wavelength of input beam. */
    struct json_object *num_inputs;             /* Number of inputs given to program */
    struct json_object *num_dipoles;            /* Number of dipoles. */
    struct json_object *max_iterations;         /* Maximum optimisation iterations. */
    struct json_object *save_positions_every_X_epochs;   /* Don't have to necessarily create a frame every epoch. */
    struct json_object *resolution_factor;      /* Factor by which to reduce resolution of evaluated fields on output. */
    struct json_object *scatterer_size;         /* Array of integers describing dimensions of scatterer. */
    struct json_object *extra_z_evaluation;     /* How much further to evaluate the field beyond the scatterer, to hopefully encompass the detector. */
    struct json_object *detector_z;             /* z-plane of 'detector'. */
    struct json_object *polarisability;         /* Polarisability of Dipoles/ */

    /* Build filepath string. */
    strcpy(filepath, "..//experiments//");
    strcat(filepath, config.experiment_name);
    strcat(filepath, "//experiment_config.json");

    /* Read entire json file into buffer. */
    in_file = fopen(filepath, "r");
    FILE_CHECK(in_file, filepath);
    fread(line_buffer, sizeof(line_buffer), 1, in_file);
    fclose(in_file);
    parsed_json = json_tokener_parse(line_buffer);

    /* Read in values from parsed json */
    json_object_object_get_ex(parsed_json, "wavelength", &wavelength);
    json_object_object_get_ex(parsed_json, "num_inputs", &num_inputs);
    json_object_object_get_ex(parsed_json, "max_iterations", &max_iterations);
    json_object_object_get_ex(parsed_json, "num_dipoles", &num_dipoles);
    json_object_object_get_ex(parsed_json, "save_positions_every_X_epochs", &save_positions_every_X_epochs);
    json_object_object_get_ex(parsed_json, "resolution_factor", &resolution_factor);
    json_object_object_get_ex(parsed_json, "scatterer_dimensions", &scatterer_size);
    json_object_object_get_ex(parsed_json, "extra_z_evaluation", &extra_z_evaluation);
    json_object_object_get_ex(parsed_json, "detector_z", &detector_z);
    json_object_object_get_ex(parsed_json, "polarisability", &polarisability);

    /* Assign read in values. */
    config.mod_k = (2.0 * M_PI) / json_object_get_int(wavelength);
    config.max_iterations = json_object_get_int(max_iterations);
    config.num_inputs = json_object_get_int(num_inputs);
    dipoles->num_dipoles = json_object_get_int(num_dipoles);
    config.save_positions_every_X_epochs = json_object_get_int(save_positions_every_X_epochs);
    config.resolution_factor = json_object_get_int(resolution_factor);
    dipoles->polarisability = json_object_get_double(polarisability);
    config.scatterer_size[X] = json_object_get_int(json_object_array_get_idx(scatterer_size, 0));
    config.scatterer_size[Z] = json_object_get_int(json_object_array_get_idx(scatterer_size, 1));
    config.detector_z = json_object_get_int(detector_z);
    config.extra_z_evaluation = json_object_get_int(extra_z_evaluation);

    return NO_ERROR;
}

static ErrorCode unpack_dipoles(Dipoles *dipoles)
{
    FILE *in_file;                          /* File containing dipoles position information. */
    char filepath[FILENAME_BUFFER];         /* Filepath toe file above. */
    char line_buffer[LINE_BUFFER_LENGTH];   /* Line buffer to read lines of file into. */
    char *line;                             /* String containing line read from file. */

    /* Build filepath string. */
    strcpy(filepath, "..//experiments//");
    strcat(filepath, config.experiment_name);
    strcat(filepath, "//initial_dipole_placement.txt");

    in_file = fopen(filepath, "r");
    FILE_CHECK(in_file, filepath);

    /* Allocate memory to store dipoles, now that we know how many there will be */
    dipoles->_dipoles = (Dipole *) malloc(dipoles->num_dipoles * sizeof(Dipole));

    /* Read all other lines from file, being dipoles positions */
    for (int i = 0; i < dipoles->num_dipoles; i++) {
        line = fgets(line_buffer, sizeof(line_buffer), in_file);
        if (feof(in_file) || (strcmp(line, "") == 0)) {
            fprintf(stderr, "Error: Premature end of file when reading from dipole placement file.\n");
            return BAD_FILE;
        }
        READLINE_CHECK(line, in_file);

        /* Assign dipoles positions by splitting line string around ',' */
        dipoles->_dipoles[i].position[X] = atoi(strtok(line, "\t"));
        dipoles->_dipoles[i].position[Z] = atoi(strtok(NULL, "\t"));

        /* Used later when iterating over dipoles. */
        dipoles->_dipoles[i]._id = i;

    }

    fclose(in_file);
    return NO_ERROR;
}

static ErrorCode unpack_input_fields(InputField **input_fields)
{
    gsl_complex E; /* Used in reading field data. */

    *input_fields = (InputField *) malloc(config.num_inputs * sizeof(InputField));

    /* TODO: Check correct number of inputs have been passed, as in json details. */

    for (int input_num = 0; input_num < config.num_inputs; input_num++) {

        FILE *in_file;                          /* File containing input field. */
        char filepath[FILENAME_BUFFER];         /* Filepath to file above. */
        char line_buffer[LINE_BUFFER_LENGTH];   /* Line buffer to read lines of file into. */
        char *line;                             /* String containing line read from file. */
        char *temp_string;
        char input_field_num_char[2];

        /* Build filepath string. */
        strcpy(filepath, "..//experiments//");
        strcat(filepath, config.experiment_name);
        strcat(filepath, "//input_fields//");
        sprintf(input_field_num_char, "%d", input_num);
        strcat(filepath, input_field_num_char);
        strcat(filepath, ".txt");

        in_file = fopen(filepath, "r");
        FILE_CHECK(in_file, filepath);

        /* Allocate memory based on X dimension on scatterer, assuming that dimension of input field matches it. */
        (*input_fields)[input_num].k_space = (gsl_complex_packed_array) malloc(config.scatterer_size[X] * sizeof(gsl_complex));

        /* Read header line, extracting focus point. */
        line = fgets(line_buffer, sizeof(line_buffer), in_file);
        READLINE_CHECK(line, in_file);

        /* Takes strtok up to the '=' character. */
        temp_string = strtok(line, "=");
        STRTOK_CHECK(temp_string, filepath, in_file);

        /* Takes strtok up to the next space, leaving the X dimension of the focus point in temp_string. */
        temp_string = strtok(NULL, " ");
        STRTOK_CHECK(temp_string, filepath, in_file);
        (*input_fields)[input_num].focus_point[X] = atoi(temp_string);

        /* Takes strtok up to the next space, leaving the Z dimension of the focus point in temp_string. */
        temp_string = strtok(NULL, " ");
        STRTOK_CHECK(temp_string, filepath, in_file);
        (*input_fields)[input_num].focus_point[Z] = atoi(temp_string);

        /* Read input field data into array. */
        for (int i = 0; i < config.scatterer_size[X]; i++) {
            line = fgets(line_buffer, sizeof(line_buffer), in_file);
            if (feof(in_file) || (strcmp(line, "") == 0)) {
                fprintf(stderr, "Error: Premature end of file when reading from input field file.\n");
                return BAD_FILE;
            }
            READLINE_CHECK(line, in_file);

            /* Read real part of field. */
            temp_string = strtok(line, "\t");
            STRTOK_CHECK(temp_string, filepath, in_file);
            GSL_SET_REAL(&E, atof(temp_string));

            /* Read imaginary part of field. */
            temp_string = strtok(NULL, "\t");
            STRTOK_CHECK(temp_string, filepath, in_file);
            GSL_SET_IMAG(&E, atof(temp_string));

            /* Assign data into k_space array, currently representing the real space version of the field. */
            GSL_SET_COMPLEX_PACKED((*input_fields)[input_num].k_space, i, GSL_REAL(E), GSL_IMAG(E));
        }

        fclose(in_file);

        /* Calculate Fourier transform of input field, a quantity more useful in the simulation. */
        gsl_fft_complex_radix2_forward((*input_fields)[input_num].k_space, 1, config.scatterer_size[X]);
    }

    return NO_ERROR;
}

static ErrorCode save_positions(Dipoles dipoles, int epoch_num, double fitness, InputField input_field)
{
    FILE *out_file;
    char filepath[FILENAME_BUFFER];
    char epoch_dir[FILENAME_BUFFER];

    if (epoch_num == -1) {
        /* Build filename. */
        strcpy(filepath, "..//experiments//");
        strcat(filepath, config.experiment_name);
        strcat(filepath, "//final_dipole_placement.txt");
    } else {
        /* Create directory. */
        strcpy(filepath, "..//experiments//");
        strcat(filepath, config.experiment_name);
        strcat(filepath, "//epochs");
        sprintf(epoch_dir, "//epoch_%d", epoch_num + 1);
        strcat(filepath, epoch_dir);
        mkdir(filepath, 0777);

        /* Build filename. */
        strcpy(filepath, "..//experiments//");
        strcat(filepath, config.experiment_name);
        strcat(filepath, "//epochs");
        strcat(filepath, epoch_dir);
        strcat(filepath, "//dipole_placement.txt");
    }
    
    out_file = fopen(filepath, "w");
    FILE_CHECK(out_file, filepath);

    /* Write fitness as file header */
    if (epoch_num != -1) {
        fprintf(out_file, "Fitness: %g\n", fitness);
    }

    /* Write positions as tab seperate coordinates. */
    for (int i = 0; i < dipoles.num_dipoles; i++) {
        fprintf(out_file, "%d\t%d\n", dipoles._dipoles[i].position[X], dipoles._dipoles[i].position[Z]);
    }

    fclose(out_file);

    return NO_ERROR;
}

static ErrorCode process_results(Dipoles dipoles, InputField *input_fields)
{
    FILE *out_file;
    char filepath[FILENAME_BUFFER];
    char input_field_num_char[2];

    /* 
     * Save final positions of dipoles.
     * An epoch_num of '-1' tells save_positions() that this is the final dipole placement
     */
    save_positions(dipoles, -1, 0.0, input_fields[0]);

    /*
     * Next, iterate over input fields saving their influence on the dipoles and
     * the detector.
     */
    for (int input_num = 0; input_num < config.num_inputs; input_num++) {

        /****** First, save field on detector. ******/

        gsl_complex_packed_array detector_field;

        /* Build filename. */
        strcpy(filepath, "..//experiments//");
        strcat(filepath, config.experiment_name);
        strcat(filepath, "//result_fields//input_");
        sprintf(input_field_num_char, "%d", input_num);
        strcat(filepath, input_field_num_char);
        strcat(filepath, "_detector_field.txt");

        /* Find field on detector. */
        detector_field = evaluate_total_field_at_plane(dipoles, input_fields[input_num], config.detector_z);

        out_file = fopen(filepath, "w");
        FILE_CHECK(out_file, filepath);

        for (int x = 0; x < config.scatterer_size[X]; x++) {
            gsl_complex E;

            /* Retreive field at point of interest. */
            GSL_SET_REAL(&E, PACKED_REAL(detector_field, x));
            GSL_SET_IMAG(&E, PACKED_IMAG(detector_field, x));

            fprintf(out_file, "%g\n", gsl_complex_abs(E));
        }

        fclose(out_file);

        free(detector_field);

        /****** Second, evaluate field around dipoles, shrunk by resolution factor. ******/

        /* Build filename. */
        strcpy(filepath, "..//experiments//");
        strcat(filepath, config.experiment_name);
        strcat(filepath, "//result_fields//input_");
        sprintf(input_field_num_char, "%d", input_num);
        strcat(filepath, input_field_num_char);
        strcat(filepath, "_lattice_field.txt");

        out_file = fopen(filepath, "w");
        FILE_CHECK(out_file, filepath);

        /* Iterate over z planes, propagating input field to that point. */
        for (int z = 0; z < config.scatterer_size[Z] + config.extra_z_evaluation; z += config.resolution_factor) {
            /* Find total field (input and dipoles) at plane. */
            gsl_complex_packed_array lattice_field = evaluate_total_field_at_plane(dipoles, input_fields[input_num], z);

            /* Iterate over points to be considered, saving them. */
            for (int x = 0; x < config.scatterer_size[X]; x += config.resolution_factor) {
                double E_real;
                E_real = PACKED_REAL(lattice_field, x);

                if isnan(E_real) { /* If dipole is at this point a div by 0 error will occur. */
                    E_real = 0.0;
                }
                fprintf(out_file, "%g ", E_real);
            }
            fprintf(out_file, "\n");

            free(lattice_field);
        }

        fclose(out_file);
    }

    return NO_ERROR;
}

/*----------------------------------PROPAGATION FUNCTIONS-------------------------------*/

static gsl_complex_packed_array evaluate_total_field_at_plane(Dipoles dipoles, InputField input_field, unsigned int z)
{
    gsl_complex_packed_array field;

    /* Find the field at the plane due to the input field only. */
    field = propagate_k_field_to_plane(input_field, z);

    /* Calculate field at point due to dipoles and add to existing field. */
    for (int i = 0; i < config.scatterer_size[X]; i++) {

        gsl_complex E;
        double rjk;
        int vector_between[DIMS];
        gsl_complex Ajk;
        int point[DIMS];

        point[X] = i;
        point[Z] = z;

        /* Retrieve current field at point of interest. */
        GSL_SET_REAL(&E, PACKED_REAL(field, i));
        GSL_SET_IMAG(&E, PACKED_IMAG(field, i));

        /* Loop over dipoles, adding contribution from each. */
        for (int d = 0; d < dipoles.num_dipoles; d++) {
            vector_between[X] = dipoles._dipoles[d].position[X] - point[X];
            vector_between[Z] = dipoles._dipoles[d].position[Z] - point[Z];
            rjk = sqrt( (vector_between[X] * vector_between[X])
                      + (vector_between[Z] * vector_between[Z]) );

            Ajk = gsl_complex_polar( (1.0 / rjk) , ( config.mod_k * rjk ) );

            E = gsl_complex_sub(E, gsl_complex_mul(Ajk, dipoles._dipoles[d].polarisation));
        }

        /* Assign total field at point into field array. */
        PACKED_SET_REAL(field, i, GSL_REAL(E));
        PACKED_SET_IMAG(field, i, GSL_IMAG(E));
    }

    return field;
}

static gsl_complex_packed_array propagate_k_field_to_plane(InputField input_field, unsigned int z)
{
    gsl_complex_packed_array out_field;
    gsl_complex k_field_value;
    int point[DIMS];

    point[Z] = z;

    out_field = (gsl_complex_packed_array) malloc(config.scatterer_size[X] * sizeof(gsl_complex));

    /* Calculate k-space field value at plane. */
    for (int x = 0; x < config.scatterer_size[X]; x++) {
        point[X] = x;

        k_field_value = propagate_k_field_to_k_point(input_field, point);

        if (isnan(gsl_complex_abs(k_field_value))) {
            k_field_value = gsl_complex_rect(0.0, 0.0);
        }

        /* Insert k field value into k_space field packed array. */
        PACKED_SET_REAL(out_field, x, GSL_REAL(k_field_value));
        PACKED_SET_IMAG(out_field, x, GSL_IMAG(k_field_value));
    }

    /* Now, find the inverse FFT of the k-space field to yield the real field at the correct plane. */
    gsl_fft_complex_radix2_inverse(out_field, 1, config.scatterer_size[X]);

    return out_field;
}

static gsl_complex propagate_k_field_to_k_point(InputField input_field, int point[DIMS])
{
    gsl_complex k_field_value;
    gsl_complex original;
    gsl_complex phase;
    double k_z;
    double k_x;
    double sqrt_arg;

    GET_KX(&k_x, config.scatterer_size[X], point[X]);

    sqrt_arg = (config.mod_k * config.mod_k) - (k_x * k_x);

    /* k_z = \sqrt{k^2 - k_x^2} */
    if (sqrt_arg > 0.0) {
        k_z = sqrt(sqrt_arg);
    } else {
        return gsl_complex_rect(0.0, 0.0);
    }

    original = gsl_complex_rect(PACKED_REAL(input_field.k_space, point[X]), PACKED_IMAG(input_field.k_space, point[X]));
    phase = gsl_complex_polar(1, -1 * point[Z] *k_z);

    k_field_value = gsl_complex_mul(original, phase);
    
    return k_field_value;
}

static gsl_complex evaluate_input_field_at_point(InputField input_field, int point[DIMS])
{
    /*
     * For now, just propagate input field to plane and return point from plane.
     * In future, consider evaluating FT at the point only, see GSL for formula on how.
     */

    gsl_complex_packed_array field_at_plane;
    gsl_complex E;

    /* Evaluate input field at the correct Z plane. */
    field_at_plane = propagate_k_field_to_plane(input_field, input_field.focus_point[Z]);

    /* Retrieve correct E field value. */
    GSL_SET_REAL(&E, PACKED_REAL(field_at_plane, point[X]));
    GSL_SET_IMAG(&E, PACKED_IMAG(field_at_plane, point[X]));

    free(field_at_plane);

    return E;
}

/*----------------------------------DDA CALCULATION FUNCTIONS---------------------------*/

static gsl_matrix_complex* find_A_inv(gsl_matrix_complex *A_inv, Dipoles dipoles)
{
    Dipole primary_dipole;
    Dipole secondary_dipole;
    gsl_complex one_over_alpha;
    gsl_complex jk_element;
    double rjk;
    int vector_between[DIMS];

    /* For LU decomposition and inverting purposes. */
    gsl_permutation *perm = gsl_permutation_calloc(dipoles.num_dipoles);
    int signum;

    one_over_alpha = gsl_complex_rect(1.0 / dipoles.polarisability, 0);

    /* First, build A matrix. */
    for (int j = 0; j < dipoles.num_dipoles; j++) {
        primary_dipole = dipoles._dipoles[j];
        for (int k = 0; k < dipoles.num_dipoles; k++) {

            /* A matrix of dipole considering itself is just 1 over the polarisability. */
            if (j == k) {
                gsl_matrix_complex_set(A_inv, j, j, one_over_alpha);
        
            } else {
                secondary_dipole = dipoles._dipoles[k];

                /* Find distance between dipoles. */
                vector_between[X] = primary_dipole.position[X] - secondary_dipole.position[X];
                vector_between[Z] = primary_dipole.position[Z] - secondary_dipole.position[Z];
                rjk = sqrt( (vector_between[X] * vector_between[X])
                          + (vector_between[Z] * vector_between[Z]) );

                jk_element = gsl_complex_polar( (1.0 / rjk), (config.mod_k * rjk) );

                gsl_matrix_complex_set(A_inv, j, k, jk_element);
            }
        }
    }

    /* Then, find the inverse (in place) by LU decomposition. */
    gsl_linalg_complex_LU_decomp(A_inv, perm, &signum);
    gsl_linalg_complex_LU_invx(A_inv, perm);

    gsl_permutation_free(perm);

    return A_inv;
}

static void find_polarisations(InputField input_field, Dipoles *dipoles, gsl_matrix_complex *A_inv)
{
    gsl_complex E;
    gsl_complex complex_1; /* Integer 1 represented in complex form. */
    gsl_complex complex_0; /* Integer 0 represented in complex form. */
    gsl_vector_complex *P_vector;
    gsl_vector_complex *E_on_Ds;

    P_vector = gsl_vector_complex_alloc(dipoles->num_dipoles);
    E_on_Ds = gsl_vector_complex_alloc(dipoles->num_dipoles);

    complex_1 = gsl_complex_polar(1.0, 0.0);
    complex_0 = gsl_complex_polar(0.0, 0.0);

    /* 
     * First, find E field values due to input field only on each dipole
     * and assign to E_on_Ds vector.
     */
    for (int i = 0; i < dipoles->num_dipoles; i++) {
        E = evaluate_input_field_at_point(input_field, dipoles->_dipoles[i].position);
        gsl_vector_complex_set(E_on_Ds, i, E);
    }

    /* Do complex matrix multiplication P = A^{-1} \times E to yield polarisations in P_vector. */
    gsl_blas_zgemv(CblasNoTrans, complex_1, A_inv, E_on_Ds, complex_0, P_vector);

    /* Assign polarisations in P_vector to dipoles. */
    for (int i = 0; i < dipoles->num_dipoles; i++) {
        dipoles->_dipoles[i].polarisation = gsl_vector_complex_get(P_vector, i);
    }

    gsl_vector_complex_free(E_on_Ds);
    gsl_vector_complex_free(P_vector);
}

/*----------------------------------OPTIMISATION FUNCTIONS------------------------------*/

static double evaluate_virtual_dipole(Dipole *virtual_dipole, Dipoles dipoles, InputField input_field, InputField *all_input_fields)
{
    // double fitness;
    double rjk;
    int vector_between[DIMS];
    gsl_complex Ajk;
    gsl_complex VD_total_field; /* Total field on the focus point with this virtual dipole considered. */

    /* Copy intensity on focus point due to input field only. */
    VD_total_field = input_field.focus_point_field_input_only;

    /* 
     * Return the intensity of the field on the focus point, due to the input fields and the dipoles.
     * 
     * All input fields have been passed such that they can be used in future evaluation functions,
     * as this is a simple implementation for now.
     */

    for (int i = 0; i < dipoles.num_dipoles; i++) {

        if (i == virtual_dipole->_id) { /* Virtual dipole as same ID as its primary dipole. */
            vector_between[X] = virtual_dipole->position[X] - input_field.focus_point[X];
            vector_between[Z] = virtual_dipole->position[Z] - input_field.focus_point[Z];
        } else {
            vector_between[X] = dipoles._dipoles[i].position[X] - input_field.focus_point[X];
            vector_between[Z] = dipoles._dipoles[i].position[Z] - input_field.focus_point[Z];
        }

        rjk = sqrt( (vector_between[X] * vector_between[X])
                  + (vector_between[Z] * vector_between[Z]) );

        Ajk = gsl_complex_polar( (1.0 / rjk) , ( config.mod_k * rjk ) );

        VD_total_field = gsl_complex_sub(VD_total_field, gsl_complex_mul(Ajk, dipoles._dipoles[i].polarisation));
    }

    return gsl_complex_abs(VD_total_field);
}

static void consider_dipole_with_input(Dipoles dipoles, Dipole *primary_dipole, InputField *all_input_fields, InputField input_field)
{
    Dipole virtual_dipole;
    int virtual_dipole_num = 0;

    gsl_complex E; /* Various uses in function */

    int vector_between[DIMS];
    gsl_complex Ajk;
    double rjk;

    virtual_dipole._id = primary_dipole->_id;
    
    /* 
    * Iterate over a cube around the primary dipole from -1 to 1 in each dimension.
    * Consider if the virtual dipole is out of bounds of the scatterer. 
    */
    for (int z = -1; z <= 1; z++) {

        /* 
        * Evaluate input field only at this z-plane, such that all dipoles 
        * on this plane for various x can use this single calculation, rather than
        * evaluating this field for all virtual dipoles.
        */
        gsl_complex_packed_array z_plane_field;
        z_plane_field = propagate_k_field_to_plane(input_field, primary_dipole->position[Z] + z);

        virtual_dipole.position[Z] = primary_dipole->position[Z] + z;

        for (int x = -1; x <= 1; x++) {

            bool in_bounds = true;

            virtual_dipole.position[X] = primary_dipole->position[X] + x;

            /* Check virtual dipole is in bounds. */
            if ( (virtual_dipole.position[X] > config.scatterer_size[X]) || (virtual_dipole.position[X] < 0) ) {
                in_bounds = false;
            } else if ( (virtual_dipole.position[Z] > config.scatterer_size[Z]) || (virtual_dipole.position[Z] < 0) ) {
                in_bounds = false;
            }

            /* If it is out of bounds, move on to the next virtual dipole. */
            if (!in_bounds) {
                primary_dipole->vdfs[virtual_dipole_num] = -DBL_MAX;
                virtual_dipole_num++;
                continue;
            }

            /* 
             * To evaluate virtual dipole, first we need to find its would-be polarisation.
             */

            /* First, retrieve E at (z, x) from complex packed array. */
            GSL_SET_REAL(&E, PACKED_REAL(z_plane_field, virtual_dipole.position[X]));
            GSL_SET_IMAG(&E, PACKED_IMAG(z_plane_field, virtual_dipole.position[X]));

            /* Next, add E field contribution due to every OTHER dipole (i.e not including primary dipole) */
            for (int i = 0; i < dipoles.num_dipoles; i++) {

                /* Ignore primary dipole. */
                if (i == primary_dipole->_id) { 
                    continue;
                }

                vector_between[X] = dipoles._dipoles[i].position[X] - virtual_dipole.position[X];
                vector_between[Z] = dipoles._dipoles[i].position[Z] - virtual_dipole.position[Z];
                rjk = sqrt( (vector_between[X] * vector_between[X])
                            + (vector_between[Z] * vector_between[Z]) );

                Ajk = gsl_complex_polar( (1.0 / rjk) , ( config.mod_k * rjk ) );

                E = gsl_complex_sub(E, gsl_complex_mul(Ajk, dipoles._dipoles[i].polarisation));
            }

            /* 
             * P_j = \alpha_j \times E_j 
             * Here, E encapsulates the contribution from the input field and all the other dipoles.
             */
            virtual_dipole.polarisation = gsl_complex_mul(E, gsl_complex_polar(dipoles.polarisability, 0));

            /* 
             * Now polarisation of the virtual dipole has been found, 
             * we can evaluate the fitness of the virtual dipole given this input field.
             * 
             * To do this, add the would-be contribution of the virtual dipole to fp_field_minus_primary.
             * And find the intensity of the resulting total field.
             */
            primary_dipole->vdfs[virtual_dipole_num] += (float) evaluate_virtual_dipole(&virtual_dipole, dipoles, input_field, all_input_fields);
            
            if (isnan(primary_dipole->vdfs[virtual_dipole_num])) {
                primary_dipole->vdfs[virtual_dipole_num] = -DBL_MAX;
            }

            virtual_dipole_num++;
        }

        free(z_plane_field);

    }
}

static void move_dipoles(Dipoles *dipoles)
{
    Dipole *primary_dipole;
    Dipole *secondary_dipole;

    for (int dipole_num = 0; dipole_num < dipoles->num_dipoles; dipole_num++) {

        primary_dipole = &(dipoles->_dipoles[dipole_num]);
        bool can_move = true;
        int optimal_vdf_index = 0;
        float optimal_vdf = -DBL_MAX;

        /* First, use vdf data to find which point the new dipole should move to. */
        /* Find index of highest fitness value in virtual dipole fitness array (vdfs) */
        for (int i = 0; i < POINTS_AROUND; i++) {
            if (primary_dipole->vdfs[i] > optimal_vdf) {
                optimal_vdf_index = i;
                optimal_vdf = primary_dipole->vdfs[i];
            }
        }

        /* 
         * Use index to extract coordinates in which dipole should move. 
         * 
         * \Delta{z} = (index % 3) - 1
         * \Delta{x} = (index / 3) - 1 (Integer divide as to ensure downward rounding)
         */
        primary_dipole->new_position[X] = primary_dipole->position[X] + ( (optimal_vdf_index % POINTS_PER_DIM) - 1 );
        primary_dipole->new_position[Z] = primary_dipole->position[Z] + ( (optimal_vdf_index / POINTS_PER_DIM) - 1 );

        /* 
         * With the most optimal new_position found, test if the dipole can actually move to that position.
         * If it cannot, it remains stationary.
         */

        /* Loop over all previously considered dipoles and compare new positions. */
        for (int j = 0; j < dipole_num; j++) {
            bool new_positions_equal = true;
            secondary_dipole = &(dipoles->_dipoles[j]);

            for (int dim = X; dim < DIMS; dim++) {
                if (primary_dipole->new_position[dim] != secondary_dipole->new_position[dim]) {
                    new_positions_equal = false;
                    break;
                }
            }

            if (new_positions_equal) {
                can_move = false;
                break;
            }
        }

        /* 
         * Check that dipole isn't moving onto the current position of a 
         * dipole yet to be considered in case that dipole does not move when considered.
         */
        if (can_move) {
            for (int j = (dipole_num + 1); j < dipoles->num_dipoles; j++) {
                bool positions_equal = true;
                secondary_dipole = &dipoles->_dipoles[j];

            for (int dim = X; dim < DIMS; dim++) {
                if (primary_dipole->new_position[dim] != secondary_dipole->position[dim]) {
                    positions_equal = false;
                    break;
                }
            }

            if (positions_equal) {
                can_move = false;
                break;
            }

            }
        }

        /* 
         * If it cannot move, make it's new position equal to it's current position such that
         * other dipoles could potentially move to the new position later in consideration.
         */
        if (!can_move) {
            primary_dipole->new_position[X] = primary_dipole->position[X];
            primary_dipole->new_position[Z] = primary_dipole->position[Z];
        }
    }

    /* 
     * Now that the permission of all the dipole movements have been checked, the dipoles can finally
     * be assigned their new positions.
     */
    for (int dipole_num = 0; dipole_num < dipoles->num_dipoles; dipole_num++) {
        primary_dipole = &(dipoles->_dipoles[dipole_num]);

        primary_dipole->position[X] = primary_dipole->new_position[X];
        primary_dipole->position[Z] = primary_dipole->new_position[Z];
    }
}

static ErrorCode evaluate_progress(Dipoles *dipoles, InputField *input_fields, int epoch_num)
{
    ErrorCode error;
    double fitness = 0.0;
    gsl_complex_packed_array field_on_detector;

    for (int input_num = 0; input_num < config.num_inputs; input_num++) {
        field_on_detector = evaluate_total_field_at_plane(*dipoles, input_fields[input_num], config.detector_z);

        /* Iterate over focus points corresponding to input fields, taking square root of intensity. */
        fitness += gsl_hypot(
            PACKED_REAL(field_on_detector, input_fields[input_num].focus_point[X]),
            PACKED_IMAG(field_on_detector, input_fields[input_num].focus_point[X])
        );
    }

    free(field_on_detector);

    fprintf(stdout, "Epoch: %d. Fitness = %g", epoch_num, fitness);

    /* Save positions of dipoles if necessary. */
    if ( (config.save_positions_every_X_epochs != 0) && (epoch_num % config.save_positions_every_X_epochs == 0) ) {
        fprintf(stdout, " *SAVING*");
        error = save_positions(*dipoles, epoch_num, fitness, input_fields[0]);
        ERROR_CHECK(error);
    }
    fprintf(stdout, "\n");

    return NO_ERROR;
}

static ErrorCode optimise_dipoles(Dipoles *dipoles, InputField *input_fields)
{
    gsl_matrix_complex *A_inv;

    Dipole *primary_dipole;     /* Primary dipole considered when looping over dipoles. */

    A_inv = gsl_matrix_complex_alloc(dipoles->num_dipoles, dipoles->num_dipoles);

    for (int epoch_num = 0; epoch_num < config.max_iterations + 1; epoch_num++) {

        A_inv = find_A_inv(A_inv, *dipoles);

        /* Set virtual dipole fitnesses to 0. */
        for (int dipole_num = 0; dipole_num < dipoles->num_dipoles; dipole_num++) {
            for (int i = 0; i < POINTS_AROUND; i++) {
                dipoles->_dipoles[dipole_num].vdfs[i] = 0.0;
            }
        }

        for (int input_num = 0; input_num < config.num_inputs; input_num++) {

            gsl_complex_packed_array field_on_detector;

            /* Find polarisations of dipoles given this input field (and assign them to dipoles structs). */
            /* P = A^{-1} E */
            find_polarisations(input_fields[input_num], dipoles, A_inv);

            /* 
             * Find the field on the focus point corressponding to the input field,
             * due to the influence of the input field only - no dipoles.
             */
            field_on_detector = propagate_k_field_to_plane(input_fields[input_num], config.detector_z);

            input_fields[input_num].focus_point_field_input_only = gsl_complex_rect(
                PACKED_REAL(field_on_detector, input_fields[input_num].focus_point[X]),
                PACKED_IMAG(field_on_detector, input_fields[input_num].focus_point[X])
            );

            free(field_on_detector);

            for (int dipole_num = 0; dipole_num < dipoles->num_dipoles; dipole_num++) {
                primary_dipole = &dipoles->_dipoles[dipole_num];  /* Point to the current dipole of interest. */

                /* Add fitnesses gained from evaluating this input field to current virtual dipole fitnesses. */
                consider_dipole_with_input(*dipoles, primary_dipole, input_fields, input_fields[input_num]);
            }
        }

        /* With the virtual dipole fitnesses found for all dipoles found, the dipoles can be moved. */
        move_dipoles(dipoles);

        evaluate_progress(dipoles, input_fields, epoch_num);
    }

    gsl_matrix_complex_free(A_inv);

    return NO_ERROR;
}

/*----------------------------------MAIN------------------------------------------------*/

int main(int argc, char *argv[])
{
    ErrorCode error;
    Dipoles dipoles;
    InputField *input_fields;

    if (argc < 2) {
        fprintf(stderr, "Error: No experiment name given.\n");
        return BAD_ARGUMENTS;
    }

    /* Store experiment name. */
    strcpy(config.experiment_name, argv[1]);

    error = parse_JSON(&dipoles);
    ERROR_CHECK(error);

    error = unpack_dipoles(&dipoles);
    ERROR_CHECK(error);

    error = unpack_input_fields(&input_fields);
    ERROR_CHECK(error);

    error = optimise_dipoles(&dipoles, input_fields);
    ERROR_CHECK(error);

    fprintf(stdout, "Processing results...\n");
    error = process_results(dipoles, input_fields);
    ERROR_CHECK(error);

    free(dipoles._dipoles);
    for (int i = 0; i < config.num_inputs; i++) {
        free(input_fields[i].k_space);
    }
    free(input_fields);

    fprintf(stdout, "Experiment complete.\n");

    return NO_ERROR;
}