#include "../files/aspen.h"

// Cifar-10 classes
char *classes[] = {"plane", "car", "bird", "cat",
           "deer", "dog", "frog", "horse", "ship", "truck"};

// Helper functions

// Get current time in seconds
double get_sec();
// Softmax function
void softmax (float *input, float *output, unsigned int num_batch, unsigned int num_elements);
// Get top 5 results from the softmax output
void get_prob_results (float* probabilities, unsigned int num);

void compare_float_arr (float *arr1, float *arr2, int size, float err)
{
    int miss = 0;
    for (int i = 0; i < size; i++)
    {
        if (fabsf (arr1[i] - arr2[i]) > err)
        {
            if (miss < 10)
                printf ("Arrays mismatch at index %d: %f != %f\n", i, arr1[i], arr2[i]);
            miss++;
        }
    }
    if (miss > 0)
        printf ("Arrays mismatched at %d/%d indices. Exiting.\n", miss, size);
    else
        printf ("Arrays match!\n");
}

void* load_arr_a (char *file_path, unsigned int size)
{
    void *input = calloc (size, 1);
    FILE *fptr = fopen(file_path, "rb");
    if (fptr != NULL)
    {
        size_t num = fread (input, sizeof(char), size, fptr);
        if (num < size)
        {
            fprintf (stderr, "Error: Failed to read file %s - Size mismatch. File size: %ld, Req. size: %d. Exiting.\n", 
                file_path, num, size);
            free (input);
            fclose(fptr);
            assert(0);
            return NULL;
        }
        fclose(fptr);
        return input;
    }
    else
    {
        fprintf(stderr, "Error: Failed to open file %s. Exiting.\n", file_path);
        free (input);
        return NULL;
    }
}

int main (int argc, char* argv[])
{
    //Take number of generation iteration from args
    int batch_size = 10;
    int num_iter = 10;
    int num_threads = sysconf(_SC_NPROCESSORS_ONLN);
    if (argc == 2)
    {
        num_iter = atoi(argv[1]);
    }
    else 
    {
        printf("Usage: ./aspen_custom <num_iter>\n");
        return 0;
    }

    print_aspen_build_info();

    // Parse DNN model specification from .cfg file and model weights from .bin file
    aspen_dnn_t *aspen_dnn = apu_create_dnn("custom.cfg", "custom_weight.bin");

    // Print the DNN model specifications. 
    // Second argument is the verbosity level for model data (weights, bias, etc.)
    // print_dnn_info (aspen_dnn, 0);

    // Save the model to a file
    apu_save_dnn_to_file (aspen_dnn, "custom.aspen");
    nasm_t *aspen_nasm = apu_generate_nasm (aspen_dnn, batch_size, num_iter);

    // // Print the NASM (ASPEN Graph) for the DNN model.
    // // Second argument is the verbosity level for Ninst (ASPEN node).
    // // Third argument is the verbosity level for data (outputs, etc.)
    // print_nasm_info (aspen_nasm, 0, 0);

    apu_save_nasm_to_file (aspen_nasm, "custom.nasm");

    // Initialize Ready Pool.
    rpool_t *rpool = rpool_init ();
    // Initialize Distributed Scheduling Engines.
    dse_group_t *dse_group = dse_group_init (num_threads);
    // Set the Ready Pool as the target for the Distributed Scheduling Engines.
    dse_group_set_rpool (dse_group, rpool);
    // Add the NASM to the Ready Pool, with input tensor data from the specified directory.
    rpool_add_nasm (rpool, aspen_nasm, "custom_input.tensor");

    // Run the DSEs.
    double start_time = get_sec();
    // Run the DSEs until the NASM is completed.
    dse_group_run_until_nasm_completion (dse_group, aspen_nasm);
    double end_time = get_sec();
    printf ("Time taken: %3.6f s\n", (end_time - start_time));
    
    // // Get the output tensor data from the NASM in a NCWH order.
    LAYER_PARAMS output_order[] = {BATCH, OUT_C, OUT_H, OUT_W};
    float *layer_output = dse_get_nasm_result (aspen_nasm, output_order);
    // Get the results.
    printf ("Predicted: ");
    for (int i = 0; i < batch_size; i++)
    {
        get_prob_results (layer_output + 10*i, 10);
    }
    printf ("\n");

    // Cleanup
    free (layer_output);
    dse_group_destroy (dse_group);
    rpool_destroy (rpool);
    apu_destroy_nasm (aspen_nasm);
    apu_destroy_dnn (aspen_dnn);
    return 0;
    return 0;
}

// Helper functions implementation

double get_sec()
{
    struct timeval now;
    gettimeofday (&now, NULL);
    return now.tv_sec + now.tv_usec*1e-6;
}

void get_prob_results (float* probabilities, unsigned int num)
{
    float max_val = -INFINITY;
    int max_idx = 0;
    for (int j = 0; j < num; j++)
    {
        if (max_val < *(probabilities + j))
        {
            max_val = *(probabilities + j);
            max_idx = j;
        }
    }
    printf ("%s ", classes[max_idx]);
}
