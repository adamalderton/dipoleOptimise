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

#define DIMS 3
#define POINTS_PER_DIM 3

#define POINTS_AROUND 27 /* 3^3 */

#define X 0
#define Y 1
#define Z 2

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

#define FIELD_ACCESS(x, y) \
    ( x*config.scatterer_size[X] + y )

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

typedef struct {
    gsl_complex polarisation;
    gsl_complex total_field_on_focus;
    int focus_point[DIMS];
    int position[DIMS];
} SourceDipole;

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
static ErrorCode parse_JSON(Dipoles *dipoles, SourceDipole **all_source_dipoles);
static ErrorCode unpack_dipoles(Dipoles *dipoles);
static ErrorCode save_positions(Dipoles dipoles, int epoch_num, double fitness);
static ErrorCode process_results(Dipoles dipoles, SourceDipole *source_dipoles);

static gsl_complex_packed_array evaluate_total_field_at_plane(Dipoles dipoles, SourceDipole source_dipole, unsigned int z);
static gsl_complex evaluate_input_field_at_point(SourceDipole source_dipole, int point[DIMS]);

static gsl_matrix_complex* find_A_inv(gsl_matrix_complex *A_inv, Dipoles dipoles);
static void find_polarisations(SourceDipole source_dipole, Dipoles *dipoles, gsl_matrix_complex *A_inv);

static double evaluate_virtual_dipole(Dipole virtual_dipole, Dipole primary_dipole, Dipoles dipoles, SourceDipole source_dipole, SourceDipole *all_source_dipoles);
static void consider_dipole_with_input(Dipoles dipoles, Dipole *primary_dipole, SourceDipole *all_source_dipoles, SourceDipole source_dipole);
static void move_dipoles(Dipoles *dipoles);
static ErrorCode evaluate_progress(Dipoles *dipoles, SourceDipole *all_source_dipoles, int epoch_num);
static ErrorCode optimise_dipoles(Dipoles *dipoles, SourceDipole *all_source_dipoles);
/***************************************************************************************/

/* Global variables. */
Config config;

/*-----------------------------------IO FUNCTIONS---------------------------------------*/

static ErrorCode parse_JSON(Dipoles *dipoles, SourceDipole **all_source_dipoles)
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
    struct json_object *polarisability;         /* Polarisability of Dipoles. */

    // struct json_object *source_dipoles;
    // struct json_object *source_dipole;
    // struct json_object *source_location;
    // struct json_object *source_focus_point;
    // struct json_object *source_mod_polarisation;

    /* Build filepath string. */
    strcpy(filepath, "..//..//experiments//");
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
    json_object_object_get_ex(parsed_json, "num_source_dipoles", &num_inputs);
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
    config.scatterer_size[X] = json_object_get_int(json_object_array_get_idx(scatterer_size, X));
    config.scatterer_size[Y] = json_object_get_int(json_object_array_get_idx(scatterer_size, Y));
    config.scatterer_size[Z] = json_object_get_int(json_object_array_get_idx(scatterer_size, Z));
    config.detector_z = json_object_get_int(detector_z);
    config.extra_z_evaluation = json_object_get_int(extra_z_evaluation);

    /* Next, handle parsing of source dipoles. */
    *all_source_dipoles = (SourceDipole *) malloc(config.num_inputs * sizeof(SourceDipole));

    // json_object_object_get_ex(parsed_json, "source_dipoles", &source_dipoles);
    // for (int source_num  = 0; source_num < config.num_inputs; source_num++) {
    //     source_dipole = json_object_array_get_idx(source_dipoles, source_num);

    //     json_object_object_get_ex(source_dipole, "position", &source_location);
    //     json_object_object_get_ex(source_dipole, "focus_point", &source_focus_point);
    //     json_object_object_get_ex(source_dipole, "mod_polarisation", &source_mod_polarisation);

    //     /* Read in position of source dipole. */
    //     (*all_source_dipoles)[source_num].position[X] = json_object_get_int(json_object_array_get_idx(source_location, X));
    //     (*all_source_dipoles)[source_num].position[Y] = json_object_get_int(json_object_array_get_idx(source_location, Y));
    //     (*all_source_dipoles)[source_num].position[Z] = json_object_get_int(json_object_array_get_idx(source_location, Z));

    //     /* Read in focus point. */
    //     (*all_source_dipoles)[source_num].focus_point[X] = json_object_get_int(json_object_array_get_idx(source_focus_point, X));
    //     (*all_source_dipoles)[source_num].focus_point[Y] = json_object_get_int(json_object_array_get_idx(source_focus_point, Y));
    //     (*all_source_dipoles)[source_num].focus_point[Z] = json_object_get_int(json_object_array_get_idx(source_focus_point, Z));

    //     /* Read in polarisation (aka E field source). */
    //     (*all_source_dipoles)[source_num].polarisation = gsl_complex_rect(
    //         json_object_get_double(source_mod_polarisation),
    //         0.0
    //     );
    // }

    (*all_source_dipoles)[0].position[X] = 512;
    (*all_source_dipoles)[0].position[Y] = 512;
    (*all_source_dipoles)[0].position[X] = -4096;
    (*all_source_dipoles)[0].focus_point[X] = 512;
    (*all_source_dipoles)[0].focus_point[Y] = 512;
    (*all_source_dipoles)[0].focus_point[Z] = 1200;
    (*all_source_dipoles)[0].polarisation = gsl_complex_rect(1.0, 0.0);

    (*all_source_dipoles)[1].position[X] = 512;
    (*all_source_dipoles)[1].position[Y] = -512;
    (*all_source_dipoles)[1].position[X] = -4096;
    (*all_source_dipoles)[1].focus_point[X] = 512;
    (*all_source_dipoles)[1].focus_point[Y] = 768;
    (*all_source_dipoles)[1].focus_point[Z] = 1200;
    (*all_source_dipoles)[1].polarisation = gsl_complex_rect(1.0, 0.0);

    (*all_source_dipoles)[2].position[X] = 512;
    (*all_source_dipoles)[2].position[Y] = 1536;
    (*all_source_dipoles)[2].position[X] = -4096;
    (*all_source_dipoles)[2].focus_point[X] = 512;
    (*all_source_dipoles)[2].focus_point[Y] = 256;
    (*all_source_dipoles)[2].focus_point[Z] = 1200;
    (*all_source_dipoles)[2].polarisation = gsl_complex_rect(1.0, 0.0);

    return NO_ERROR;
}

static ErrorCode unpack_dipoles(Dipoles *dipoles)
{
    FILE *in_file;                          /* File containing dipoles position information. */
    char filepath[FILENAME_BUFFER];         /* Filepath toe file above. */
    char line_buffer[LINE_BUFFER_LENGTH];   /* Line buffer to read lines of file into. */
    char *line;                             /* String containing line read from file. */

    /* Build filepath string. */
    strcpy(filepath, "..//..//experiments//");
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
        dipoles->_dipoles[i].position[Y] = atoi(strtok(NULL, "\t"));
        dipoles->_dipoles[i].position[Z] = atoi(strtok(NULL, "\t"));

        /* Used later when iterating over dipoles. */
        dipoles->_dipoles[i]._id = i;

    }

    fclose(in_file);
    return NO_ERROR;
}

static ErrorCode save_positions(Dipoles dipoles, int epoch_num, double fitness)
{
    FILE *out_file;
    char filepath[FILENAME_BUFFER];
    char epoch_dir[FILENAME_BUFFER];

    if (epoch_num == -1) {
        /* Build filename. */
        strcpy(filepath, "..//..//experiments//");
        strcat(filepath, config.experiment_name);
        strcat(filepath, "//final_dipole_placement.txt");
    } else {
        /* Create directory. */
        strcpy(filepath, "..//..//experiments//");
        strcat(filepath, config.experiment_name);
        strcat(filepath, "//epochs");
        sprintf(epoch_dir, "//epoch_%d", epoch_num + 1);
        strcat(filepath, epoch_dir);
        mkdir(filepath, 0777);

        /* Build filename. */
        strcpy(filepath, "..//..//experiments//");
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
        fprintf(out_file, "%d\t%d\t%d\n", dipoles._dipoles[i].position[X], dipoles._dipoles[i].position[Y], dipoles._dipoles[i].position[Z]);
    }

    fclose(out_file);

    return NO_ERROR;
}

static ErrorCode process_results(Dipoles dipoles, SourceDipole *source_dipoles)
{
    FILE *out_file;
    char filepath[FILENAME_BUFFER];
    char input_field_num_char[2];

    /* 
     * Save final positions of dipoles.
     * An epoch_num of '-1' tells save_positions() that this is the final dipole placement
     */
    save_positions(dipoles, -1, 0.0);

    /*
     * Next, iterate over input fields saving their influence on the dipoles and
     * the detector.
     */
    for (int input_num = 0; input_num < config.num_inputs; input_num++) {

        gsl_complex_packed_array detector_field;

        /* Build filename. */
        strcpy(filepath, "..//..//experiments//");
        strcat(filepath, config.experiment_name);
        strcat(filepath, "//result_fields//input_");
        sprintf(input_field_num_char, "%d", input_num);
        strcat(filepath, input_field_num_char);
        strcat(filepath, "_detector_field.txt");

        /* Find field on detector. */
        detector_field = evaluate_total_field_at_plane(dipoles, source_dipoles[input_num], config.detector_z);

        out_file = fopen(filepath, "w");
        FILE_CHECK(out_file, filepath);

        for (int x = 0; x < config.scatterer_size[X]; x++) {
            for (int y = 0; y < config.scatterer_size[Y]; y++) {
                gsl_complex E;

                /* Retrieve field at point of interest. */
                GSL_SET_REAL(&E, PACKED_REAL(detector_field, FIELD_ACCESS(x, y)));
                GSL_SET_IMAG(&E, PACKED_IMAG(detector_field, FIELD_ACCESS(x, y)));

                fprintf(out_file, "%g ", gsl_complex_abs(E));
            }
            fprintf(out_file, "\n");
        }

        fclose(out_file);

        free(detector_field);
    }

    return NO_ERROR;
}

/*----------------------------------PROPAGATION FUNCTIONS-------------------------------*/

static gsl_complex_packed_array evaluate_total_field_at_plane(Dipoles dipoles, SourceDipole source_dipole, unsigned int z)
{
    gsl_complex_packed_array field;
    gsl_complex E;
    double rjk;
    int vector_between[DIMS];
    gsl_complex Ajk;
    int point[DIMS];

    field = (gsl_complex_packed_array) malloc(config.scatterer_size[X] * config.scatterer_size[Y] * sizeof(gsl_complex));

    point[Z] = z;

    /* Calculate field at point due to dipoles and source dipole. */
    for (int x = 0; x < config.scatterer_size[X]; x++) {
        point[X] = x;
        
        for (int y = 0; y < config.scatterer_size[Y]; y++) {
            point[Y] = y;

            /* Find field due to source dipole only, storing the result in E. */
            E = evaluate_input_field_at_point(source_dipole, point);

            /* Loop over dipoles, adding contribution from each. */
            for (int d = 0; d < dipoles.num_dipoles; d++) {
                vector_between[X] = dipoles._dipoles[d].position[X] - point[X];
                vector_between[Y] = dipoles._dipoles[d].position[Y] - point[Y]; 
                vector_between[Z] = dipoles._dipoles[d].position[Z] - point[Z];
                rjk = sqrt( (vector_between[X] * vector_between[X])
                          + (vector_between[Y] * vector_between[Y]) 
                          + (vector_between[Z] * vector_between[Z]) );

                Ajk = gsl_complex_polar( (1.0 / rjk) , ( config.mod_k * rjk ) );

                E = gsl_complex_sub(E, gsl_complex_mul(Ajk, dipoles._dipoles[d].polarisation));
            }

            /* Assign total field at point into field array. */
            PACKED_SET_REAL(field, FIELD_ACCESS(x, y), GSL_REAL(E));
            PACKED_SET_IMAG(field, FIELD_ACCESS(x, y), GSL_IMAG(E));
        }
    }

    return field;
}

static gsl_complex evaluate_input_field_at_point(SourceDipole source_dipole, int point[DIMS])
{
    gsl_complex E;
    int vector_between[DIMS];
    gsl_complex Ajk;
    double r;

    /* Find displacement vector between source dipole and necessary point. */
    vector_between[X] = point[X] - source_dipole.position[X];
    vector_between[Y] = point[Y] - source_dipole.position[Y];
    vector_between[Z] = point[Z] - source_dipole.position[Z];

    /* Find magnitude of distance between points. */
    r = sqrt( (vector_between[X] * vector_between[X])
            + (vector_between[Y] * vector_between[Y])
            + (vector_between[Z] * vector_between[Z]) );

    /* Find Ajk, factor by which to multiply the polarisation of the dipole. */
    //Ajk = gsl_complex_polar( (1.0 / r) , (config.mod_k * r) );
    Ajk = gsl_complex_polar(1.0, config.mod_k * r);

    /* E = Ajk * P */
    E = gsl_complex_polar(0.0, 0.0);
    E = gsl_complex_sub(E, gsl_complex_mul(Ajk, source_dipole.polarisation));

    /* Correct imaginary part of E. */
    GSL_SET_IMAG(&E, -GSL_IMAG(E));

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
                vector_between[Y] = primary_dipole.position[Y] - secondary_dipole.position[Y];
                vector_between[Z] = primary_dipole.position[Z] - secondary_dipole.position[Z];
                rjk = sqrt( (vector_between[X] * vector_between[X])
                          + (vector_between[Y] * vector_between[Y])
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

static void find_polarisations(SourceDipole source_dipole, Dipoles *dipoles, gsl_matrix_complex *A_inv)
{
    gsl_complex E;
    gsl_vector_complex *P_vector;
    gsl_vector_complex *E_on_Ds;

    P_vector = gsl_vector_complex_alloc(dipoles->num_dipoles);
    E_on_Ds = gsl_vector_complex_alloc(dipoles->num_dipoles);

    /* 
     * First, find E field values due to input field only on each dipole
     * and assign to E_on_Ds vector.
     */
    for (int i = 0; i < dipoles->num_dipoles; i++) {
        E = evaluate_input_field_at_point(source_dipole, dipoles->_dipoles[i].position);
        gsl_vector_complex_set(E_on_Ds, i, E);
    }

    /* Do complex matrix multiplication P = A^{-1} \times E to yield polarisations in P_vector. */
    gsl_blas_zgemv(CblasNoTrans, gsl_complex_polar(1.0, 0.0), A_inv, E_on_Ds, gsl_complex_polar(0.0, 0.0), P_vector);

    /* Assign polarisations in P_vector to dipoles. */
    for (int i = 0; i < dipoles->num_dipoles; i++) {
        dipoles->_dipoles[i].polarisation = gsl_vector_complex_get(P_vector, i);
    }

    gsl_vector_complex_free(E_on_Ds);
    gsl_vector_complex_free(P_vector);
}

/*----------------------------------OPTIMISATION FUNCTIONS------------------------------*/

static double evaluate_virtual_dipole(Dipole virtual_dipole, Dipole primary_dipole, Dipoles dipoles, SourceDipole source_dipole, SourceDipole *all_source_dipoles)
{
    // double fitness;
    double rjk;
    int vector_between[DIMS];
    gsl_complex Ajk;
    gsl_complex VD_total_field; /* Total field on the focus point with this virtual dipole considered. */

    VD_total_field = source_dipole.total_field_on_focus;

    /* Subtract contribution from primary dipole. */
    vector_between[X] = primary_dipole.position[X] - source_dipole.focus_point[X];
    vector_between[Y] = primary_dipole.position[Y] - source_dipole.focus_point[Y];
    vector_between[Z] = primary_dipole.position[Z] - source_dipole.focus_point[Z];

    rjk = sqrt( (vector_between[X] * vector_between[X])
              + (vector_between[Y] * vector_between[Y])
              + (vector_between[Z] * vector_between[Z]) );

    Ajk = gsl_complex_polar( (1.0 / rjk) , ( config.mod_k * rjk ) );

    VD_total_field = gsl_complex_add(VD_total_field, gsl_complex_mul(Ajk, primary_dipole.polarisation));


    /* Add contribution from virtual dipole. */
    vector_between[X] = virtual_dipole.position[X] - source_dipole.focus_point[X];
    vector_between[Y] = virtual_dipole.position[Y] - source_dipole.focus_point[Y];
    vector_between[Z] = virtual_dipole.position[Z] - source_dipole.focus_point[Z];

    rjk = sqrt( (vector_between[X] * vector_between[X])
              + (vector_between[Y] * vector_between[Y])
              + (vector_between[Z] * vector_between[Z]) );

    Ajk = gsl_complex_polar( (1.0 / rjk) , ( config.mod_k * rjk ) );

    VD_total_field = gsl_complex_sub(VD_total_field, gsl_complex_mul(Ajk, virtual_dipole.polarisation));

    return log(gsl_complex_abs(VD_total_field));
}

static void consider_dipole_with_input(Dipoles dipoles, Dipole *primary_dipole, SourceDipole *all_source_dipoles, SourceDipole source_dipole)
{
    Dipole virtual_dipole;
    int virtual_dipole_num = 0;

    gsl_complex E; /* Various uses in function */

    int vector_between[DIMS];
    gsl_complex Ajk;
    double rjk;
    bool in_bounds;

    virtual_dipole._id = primary_dipole->_id;
    
    /* 
    * Iterate over a cube around the primary dipole from -1 to 1 in each dimension.
    * Consider if the virtual dipole is out of bounds of the scatterer. 
    */
    for (int z = -1; z <= 1; z++) {
        virtual_dipole.position[Z] = primary_dipole->position[Z] + z;

        for (int y = -1; y <= 1; y++) {
            virtual_dipole.position[Y] = primary_dipole->position[Y] + y;

            for (int x = -1; x <= 1; x++) {
                virtual_dipole.position[X] = primary_dipole->position[X] + x;

                in_bounds = true;

                /* Check virtual dipole is within the bounds of the scatterer. */
                for (int dim = X; dim < DIMS; dim++) {
                    if ( (virtual_dipole.position[dim] >= config.scatterer_size[dim]) || (virtual_dipole.position[dim] < 0) ) {
                        in_bounds = false;
                        break;
                    }
                }

                if (!in_bounds) {
                    primary_dipole->vdfs[virtual_dipole_num] = -DBL_MAX;
                    virtual_dipole_num++;
                    continue;
                }

                /* 
                 * To evaluate virtual dipole, first we need to find its would-be polarisation.
                 */

                /* First, retrieve E at (x, y, z) at position of virtual dipole. */
                E = evaluate_input_field_at_point(source_dipole, virtual_dipole.position);

                /* Next, add E field contribution from every OTHER dipole. (i.e, not the primary dipole.). */
                for (int i = 0; i < dipoles.num_dipoles; i++) {

                    /* Ignore primary dipole. */
                    if (i == primary_dipole->_id) {
                        continue;
                    }

                    vector_between[X] = dipoles._dipoles[i].position[X] - virtual_dipole.position[X];
                    vector_between[Y] = dipoles._dipoles[i].position[Y] - virtual_dipole.position[Y];
                    vector_between[Z] = dipoles._dipoles[i].position[Z] - virtual_dipole.position[Z];
                    rjk = sqrt( (vector_between[X] * vector_between[X]) 
                              + (vector_between[Y] * vector_between[Y])
                              + (vector_between[Z] * vector_between[Z]) );

                    Ajk = gsl_complex_polar( (1.0 / rjk) , (config.mod_k * rjk) );
                    
                    E = gsl_complex_sub(E, gsl_complex_mul(Ajk, dipoles._dipoles[i].polarisation));
                }

                /*
                 * P_j = \alpha_j \times E_j
                 * Here, E encapsulates the contribtuion from the input field and all other dipoles.
                 */
                virtual_dipole.polarisation = gsl_complex_mul(E, gsl_complex_polar(dipoles.polarisability, 0.0));

                /* 
                * Now polarisation of the virtual dipole has been found, 
                * we can evaluate the fitness of the virtual dipole given this input field.
                * 
                * To do this, add the would-be contribution of the virtual dipole to fp_field_minus_primary.
                * And find the intensity of the resulting total field.
                */
                primary_dipole->vdfs[virtual_dipole_num] += (float) evaluate_virtual_dipole(virtual_dipole, *primary_dipole, dipoles, source_dipole, all_source_dipoles);
            
                if (isnan(primary_dipole->vdfs[virtual_dipole_num])) {
                    primary_dipole->vdfs[virtual_dipole_num] = -DBL_MAX;
                }

                virtual_dipole_num++;
            }
        }
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
         * Integer divide is used as to ensure downward rounding.
         * 
         * \Delta{z} = (index / 9) - 1
         * \Delta{y} = ((index % 9) / 3) - 1
         * \Detla{x} - (index % 3) - 1
         * 
         */
        primary_dipole->new_position[Z] = primary_dipole->position[Z] + (optimal_vdf_index / (POINTS_PER_DIM*POINTS_PER_DIM)) - 1;
        primary_dipole->new_position[Y] = primary_dipole->position[Y] + ((optimal_vdf_index % (POINTS_PER_DIM*POINTS_PER_DIM)) / POINTS_PER_DIM) - 1;
        primary_dipole->new_position[X] = primary_dipole->position[X] + (optimal_vdf_index % POINTS_PER_DIM) - 1;

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
            primary_dipole->new_position[Y] = primary_dipole->position[Y];
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
        primary_dipole->position[Y] = primary_dipole->new_position[Y];
        primary_dipole->position[Z] = primary_dipole->new_position[Z];
    }
}

static ErrorCode evaluate_progress(Dipoles *dipoles, SourceDipole *all_source_dipoles, int epoch_num)
{
    ErrorCode error;
    int focus_point[DIMS];
    double fitness = 0.0;
    gsl_complex_packed_array field_on_detector;

    for (int input_num = 0; input_num < config.num_inputs; input_num++) {
        field_on_detector = evaluate_total_field_at_plane(*dipoles, all_source_dipoles[input_num], config.detector_z);

        focus_point[X] = all_source_dipoles[input_num].focus_point[X];
        focus_point[Y] = all_source_dipoles[input_num].focus_point[Y];
        focus_point[Z] = all_source_dipoles[input_num].focus_point[Z];

        /* Iterate over focus points corresponding to input fields, taking square root of intensity. */
        fitness += log(gsl_hypot(
            PACKED_REAL(field_on_detector, FIELD_ACCESS(focus_point[X], focus_point[Y])),
            PACKED_IMAG(field_on_detector, FIELD_ACCESS(focus_point[X], focus_point[Y]))
        ));
        
        free(field_on_detector);
    }

    fprintf(stdout, "Epoch: %d. Fitness = %.8f", epoch_num, fitness);

    /* Save positions of dipoles if necessary. */
    if ( (config.save_positions_every_X_epochs != 0) && (epoch_num % config.save_positions_every_X_epochs == 0) ) {
        fprintf(stdout, " *SAVING*");
        error = save_positions(*dipoles, epoch_num, fitness);
        ERROR_CHECK(error);
    }
    fprintf(stdout, "\n");

    return NO_ERROR;
}

static ErrorCode optimise_dipoles(Dipoles *dipoles, SourceDipole *all_source_dipoles)
{
    gsl_matrix_complex *A_inv;
    Dipole *primary_dipole;     /* Primary dipole considered when looping over dipoles. */

    int vector_between[DIMS];
    gsl_complex Ajk;
    double rjk;

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

            SourceDipole *active_source;

            active_source = &all_source_dipoles[input_num];

            /* Find polarisations of dipoles given this input field (and assign them to dipoles structs). */
            /* P = A^{-1} E */
            find_polarisations(all_source_dipoles[input_num], dipoles, A_inv);

            /* Find the total field on the focus point, to prevent unneccessary looping later. */
            active_source->total_field_on_focus = evaluate_input_field_at_point(*active_source, active_source->focus_point);
            for (int d = 0; d < dipoles->num_dipoles; d++) {
                vector_between[X] = active_source->focus_point[X] - dipoles->_dipoles[d].position[X];
                vector_between[Y] = active_source->focus_point[Y] - dipoles->_dipoles[d].position[Y];
                vector_between[Z] = active_source->focus_point[Z] - dipoles->_dipoles[d].position[Z];

                rjk = sqrt( (vector_between[X] * vector_between[X]) 
                          + (vector_between[Y] * vector_between[Y])
                          + (vector_between[Z] * vector_between[Z]) );

                Ajk = gsl_complex_polar( (1.0 / rjk) , (config.mod_k * rjk) );

                active_source->total_field_on_focus = gsl_complex_sub(active_source->total_field_on_focus, gsl_complex_mul(Ajk, dipoles->_dipoles[d].polarisation));
            }

            /* Consider all dipoles. */
            for (int dipole_num = 0; dipole_num < dipoles->num_dipoles; dipole_num++) {
                primary_dipole = &dipoles->_dipoles[dipole_num];  /* Point to the current dipole of interest. */

                /* Add fitnesses gained from evaluating this input field to current virtual dipole fitnesses. */
                consider_dipole_with_input(*dipoles, primary_dipole, all_source_dipoles, all_source_dipoles[input_num]);
            }
        }

        /* With the virtual dipole fitnesses found for all dipoles found, the dipoles can be moved. */
        move_dipoles(dipoles);

        evaluate_progress(dipoles, all_source_dipoles, epoch_num);
    }

    gsl_matrix_complex_free(A_inv);

    return NO_ERROR;
}

/*----------------------------------MAIN------------------------------------------------*/

int main(int argc, char *argv[])
{
    ErrorCode error;
    Dipoles dipoles;
    SourceDipole *all_source_dipoles;

    if (argc < 2) {
        fprintf(stderr, "Error: No experiment name given.\n");
        return BAD_ARGUMENTS;
    }

    /* Store experiment name. */
    strcpy(config.experiment_name, argv[1]);

    error = parse_JSON(&dipoles, &all_source_dipoles);
    ERROR_CHECK(error);

    error = unpack_dipoles(&dipoles);
    ERROR_CHECK(error);

    fprintf(stdout, "Beginning optimisation process...\n");
    error = optimise_dipoles(&dipoles, all_source_dipoles);
    ERROR_CHECK(error);

    fprintf(stdout, "Processing results...\n");
    error = process_results(dipoles, all_source_dipoles);
    ERROR_CHECK(error);

    free(dipoles._dipoles);
    free(all_source_dipoles);

    fprintf(stdout, "Experiment complete.\n");

    return NO_ERROR;
}