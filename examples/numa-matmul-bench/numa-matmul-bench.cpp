#include "common.h"
#include "ggml.h"

#include <locale.h>
#include <assert.h>
#include <math.h>
#include <cstring>
#include <cstdio>
#include <cinttypes>
#include <unordered_map>
#include <queue>
#include <string.h>
#include <cassert>
#include <fstream>
#include <string>
#include <iterator>
#include <algorithm>
#include <vector>
#include <stdint.h>
#include <unistd.h>
#include <numaif.h>
#include <signal.h>
#include <numa.h>
#include <sys/mman.h>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

int get_numa_nodes(void * buf, size_t buf_len, int * nodes, size_t nodes_len);
void segv_handler(int sig, siginfo_t *si, void *ctx);

int get_numa_nodes(void * buf, size_t buf_len, int * nodes, size_t nodes_len) {
    if (buf_len == 0) {
        return 0;
    }

    long page_size = sysconf(_SC_PAGESIZE);
    if (page_size == -1) {
        return -1;
    }

    uintptr_t buf_addr = (uintptr_t)buf;
    uintptr_t page_mask = ~(page_size - 1UL);
    uintptr_t page_start = buf_addr & page_mask;
    uintptr_t end_addr = buf_addr + buf_len;
    uintptr_t end_page_start = (end_addr - 1) & page_mask;

    if (end_page_start < page_start) {
        // Handle overflow (unlikely but possible with very high addresses)
        return -1;
    }

    size_t num_pages = ((end_page_start - page_start) / page_size) + 1;

    if (nodes == NULL || nodes_len < num_pages) {
        return -1;
    }

    void ** pages = (void**) malloc(num_pages * sizeof(void *));
    if (!pages) {
        return -1;
    }

    int * status = (int*) malloc(num_pages * sizeof(int));
    if (!status) {
        free(pages);
        return -1;
    }

    for (size_t i = 0; i < num_pages; i++) {
        pages[i] = (void *)(page_start + i * page_size);
    }

    int ret = move_pages(0, num_pages, pages, NULL, status, 0);
    if (ret < 0) {
        free(pages);
        free(status);
        return -1;
    }

    for (size_t i = 0; i < num_pages; i++) {
        if (status[i] < 0) {
            printf("status[%ld] = %d\n", i, status[i]);
            free(pages);
            free(status);
            return -1;
        }
        nodes[i] = status[i];
    }

    free(pages);
    free(status);

    return num_pages;
}

static void ggml_graph_compute_helper(std::vector<uint8_t> & buf, ggml_cgraph * graph, int n_threads) {
    struct ggml_cplan plan = ggml_graph_plan(graph, n_threads, nullptr);

    if (plan.work_size > 0) {
        buf.resize(plan.work_size);
        plan.work_data = buf.data();
    }

    ggml_graph_compute(graph, &plan);
}

static float tensor_sum_elements(const ggml_tensor * tensor) {
    double sum = 0;
    if (tensor->type == GGML_TYPE_F32) {
        for (int j = 0; j < tensor->ne[1]; j++) {
            for (int k = 0; k < tensor->ne[0]; k++) {
                sum += ((float *) tensor->data)[j*tensor->ne[0] + k];
            }
        }
    }
    return sum;
}

static void tensor_dump(const ggml_tensor * tensor, const char * name) {
    printf("%15s: type = %i (%5s) ne = %5" PRIi64 " x %5" PRIi64 " x %5" PRIi64 ", nb = (%5zi, %5zi, %5zi) - ", name,
        tensor->type, ggml_type_name(tensor->type),
        tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->nb[0], tensor->nb[1], tensor->nb[2]);
    float sum = tensor_sum_elements(tensor);
    printf("Sum of tensor %s is %6.2f\n", name, sum);
}

#define TENSOR_DUMP(tensor) tensor_dump(tensor, #tensor)

struct benchmark_params_struct {
    int     n_threads      = 1;
    int32_t n_iterations   = 10;
    int32_t n_layers       = 100;
    int     n_experts      = 1;
    int     n_experts_used = 1;
    int     ne0            = 8192;
    int     ne1            = 2048;
    int     ne2            = 1;
    enum ggml_numa_strategy numa = GGML_NUMA_STRATEGY_DISABLED;
    bool    mmap           = false;
    bool    verbose        = false;
};

static void print_usage(int /*argc*/, char ** argv, struct benchmark_params_struct params) {
    fprintf(stderr, "usage: %s [options]\n", argv[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h, --help            show this help message and exit\n");
    fprintf(stderr, "  -t N, --threads N     number of threads to use during computation (default: %d)\n", params.n_threads);
    fprintf(stderr, "  -i N, --iter N        number of iterations to use during computation (default: %d)\n", params.n_iterations);
    fprintf(stderr, "  -l N, --layers N      number of matmul layers (default: %d)\n", params.n_layers);
    fprintf(stderr, "  -m,   --mmap          allocate matrices with mmap buffer type\n");
    fprintf(stderr, "  -n TYPE, --numa TYPE  NUMA strategy (default: none)\n");
    fprintf(stderr, "  -e N, --experts N     number of experts (default %d)\n", params.n_experts);
    fprintf(stderr, "  -x N                  first dimension of the tensor (default %d)\n", params.ne0);
    fprintf(stderr, "  -y N                  second dimension of the tensor (default %d)\n", params.ne1);
    fprintf(stderr, "  -z N                  third dimension of the tensor (default %d)\n", params.ne2);
    fprintf(stderr, "  -v,   --verbose       increase output verbosity\n");
    fprintf(stderr, "\n");
}

#define PAGE_SIZE 4096

// SIGSEGV handler (runs in the faulting thread's context)
void segv_handler(int sig, siginfo_t *si, void *ctx) {
    GGML_UNUSED(sig);
    GGML_UNUSED(ctx);

    void *addr = si->si_addr;
    void *page_start = (void *)((uintptr_t)addr & ~(PAGE_SIZE-1));

    // Make the page writable and perform a dummy write (triggers allocation)
    mprotect(page_start, PAGE_SIZE, PROT_READ | PROT_WRITE);
    *((volatile char *)page_start) = *((volatile char *)page_start); // Dummy write
}

int main(int argc, char ** argv)  {
    struct benchmark_params_struct benchmark_params;

    bool invalid_param = false;
    std::string arg;
    for (int i = 1; i < argc; i++) {
        arg = argv[i];

        if (arg == "-t" || arg == "--threads") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            benchmark_params.n_threads = std::stoi(argv[i]);
        } else if (arg == "-i" || arg == "--iter") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            benchmark_params.n_iterations = std::stoi(argv[i]);
        } else if (arg == "-l" || arg == "--layers") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            benchmark_params.n_layers = std::stoi(argv[i]);
        } else if (arg == "-e" || arg == "--experts") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            benchmark_params.n_experts = std::stoi(argv[i]);
        } else if (arg == "-u" || arg == "--used-experts") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            benchmark_params.n_experts_used = std::stoi(argv[i]);
        } else if (arg == "-x") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            benchmark_params.ne0 = std::stoi(argv[i]);
        } else if (arg == "-y") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            benchmark_params.ne1 = std::stoi(argv[i]);
        } else if (arg == "-z") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            benchmark_params.ne2 = std::stoi(argv[i]);
        } else if (arg == "-m" || arg == "--mmap") {
            benchmark_params.mmap = true;
        } else if (arg == "-n" || arg == "--numa") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            if (std::string(argv[i]) == "distribute") {
                benchmark_params.numa = GGML_NUMA_STRATEGY_DISTRIBUTE;
            } else if (std::string(argv[i]) == "isolate") {
                benchmark_params.numa = GGML_NUMA_STRATEGY_ISOLATE;
            } else if (std::string(argv[i]) == "numactl") {
                benchmark_params.numa = GGML_NUMA_STRATEGY_NUMACTL;
            } else {
                invalid_param = true;
                break;
            }
        }  else if (arg == "-h" || arg == "--help") {
            print_usage(argc, argv, benchmark_params);
            exit(0);
        } else if (arg == "-v" || arg == "--verbose") {
            benchmark_params.verbose = true;
        }
    }
    if (invalid_param) {
        fprintf(stderr, "error: invalid parameter for argument: %s\n", arg.c_str());
        print_usage(argc, argv, benchmark_params);
        exit(1);
    }

    print_build_info();
    printf("Starting Test\n");

    if (benchmark_params.mmap) {
        // Install SIGSEGV handler
        struct sigaction sa;
        sa.sa_sigaction = segv_handler;
        sa.sa_flags = SA_SIGINFO;

        sigaction(SIGSEGV, &sa, NULL);
    }

    // benchmark parameters
    int n_layers = benchmark_params.n_layers;
    int n_experts = benchmark_params.n_experts;
    int n_experts_used = benchmark_params.n_experts_used;
    GGML_UNUSED(n_experts_used);

    // create the ggml context
    struct ggml_context * ctx;
    struct ggml_context * ctx2;

#undef VERBOSE_DEBUGGING
#ifndef VERBOSE_DEBUGGING
    const int sizey = benchmark_params.ne1;
    const int sizex = benchmark_params.ne0;
    const int sizez = benchmark_params.ne2;
#else
    /* Working - let's increase size */
    const int sizey = 1;
    const int sizex = (8*32);
    const int sizez = 1;
#endif

    //printf("Memsize required = %i\n", sizex*sizex);

    // TODO: perform the bench for all types or for a user specified type
    const ggml_type qtype = GGML_TYPE_Q4_1;

    size_t ctx_size = 0;
    ctx_size += (3 + 4 * n_layers) * ggml_tensor_overhead();
    ctx_size += 2 * ggml_graph_overhead();

    size_t ctx_size2 = 0;
    ctx_size2 += (2 * n_layers) * ggml_tensor_overhead();

    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ NULL,
        /* no_alloc   =*/ true
    };

    struct ggml_init_params params2 = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ NULL,
        /* no_alloc   =*/ true
    };

    ctx = ggml_init(params);
    if (!ctx) {
        fprintf(stderr, "%s: ggml_init() failed\n", __func__);
        return 1;
    }

    ctx2 = ggml_init(params2);
    if (!ctx2) {
        fprintf(stderr, "%s: ggml_init() failed\n", __func__);
        return 1;
    }

    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(backend, benchmark_params.n_threads);

    if (benchmark_params.numa != GGML_NUMA_STRATEGY_DISABLED) {
        auto * dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        GGML_ASSERT(dev && "CPU backend is not loaded");
        auto * reg = ggml_backend_dev_backend_reg(dev);
        auto * numa_init_fn = (decltype(ggml_numa_init) *) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_numa_init");
        numa_init_fn(benchmark_params.numa);
    }

    ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();
    ggml_backend_buffer_type_t buft2 = benchmark_params.mmap ? ggml_backend_mmap_buffer_type() : ggml_backend_cpu_buffer_type();

    // create tensors
    struct ggml_tensor * m2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, sizex, sizez);
    struct ggml_tensor * cur = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, sizey, sizez);
    struct ggml_tensor * ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_experts);
    struct ggml_tensor * cur1 = cur;
    struct ggml_tensor * cur2 = cur;
    std::vector<struct ggml_tensor *> m11s(n_layers);
    std::vector<struct ggml_tensor *> q11s(n_layers);
    for (int l = 0; l < n_layers; ++l) {
        struct ggml_tensor * mul1 = nullptr;
        struct ggml_tensor * mul2 = nullptr;

        if (n_experts > 1) {
            m11s[l] = ggml_new_tensor_3d(ctx2, GGML_TYPE_F32, sizex, sizey, n_experts);
            q11s[l] = ggml_new_tensor_3d(ctx2, qtype, sizex, sizey, n_experts);

            mul1 = ggml_mul_mat_id(ctx, m11s[l], m2, ids);
            mul2 = ggml_mul_mat_id(ctx, q11s[l], m2, ids);
        } else {
            m11s[l] = ggml_new_tensor_2d(ctx2, GGML_TYPE_F32, sizex, sizey);
            q11s[l] = ggml_new_tensor_2d(ctx2, qtype, sizex, sizey);

            mul1 = ggml_mul_mat(ctx, m11s[l], m2);
            mul2 = ggml_mul_mat(ctx, q11s[l], m2);
        }

        cur1 = ggml_add(ctx, mul1, cur1);
        cur2 = ggml_add(ctx, mul2, cur2);
    }

    // alloc tensors
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    ggml_backend_buffer_t buffer2 = ggml_backend_alloc_ctx_tensors_from_buft(ctx2, buft2);

    GGML_ASSERT(ggml_backend_buffer_is_host(ids->buffer));
    int32_t * data = (int32_t *) ids->data;
    for (int i = 0; i < n_experts; ++i) {
        data[i] = i;
    }

    // create graphs
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, cur1);
    struct ggml_cgraph * gf31 = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf31, cur2);

    // compute graphs to allocate mmapped tensors
    ggml_backend_graph_compute(backend, gf);
    ggml_backend_graph_compute(backend, gf31);

    // examine NUMA nodes owning tensor buffer pages
    if (benchmark_params.mmap && benchmark_params.verbose) {
        #define NUM_NODES 65536
        int nodes[NUM_NODES];
        for (int l = 0; l < n_layers; ++l) {
            void * buf = q11s[l]->data;
            size_t buf_len = q11s[l]->nb[3];

            printf("checking tensor with buf = %p, buf len = %ld, row len %ld bytes\n", buf, buf_len, q11s[l]->nb[1]);
            int num_buf_nodes = get_numa_nodes(buf, buf_len, nodes, NUM_NODES);
            printf("num_buf_nodes = %d\n", num_buf_nodes);
            if (num_buf_nodes < 0) {
                printf("get_numa_nodes failed for layer %d\n", l);
            } else {
                int current_node = nodes[0];
                int current_node_ctr = 0;
                int num_printed = 0;
                int num_row = 0;
                for (int n = 0; n < num_buf_nodes; ++n) {
                    uintptr_t page_start = (uintptr_t) buf + PAGE_SIZE * n;
                    GGML_UNUSED(page_start);
                    uintptr_t page_end = (uintptr_t) buf + PAGE_SIZE * (n + 1);
                    for (; num_row < q11s[l]->ne[1]; ++num_row) {
                        uintptr_t row_start = (uintptr_t) buf + q11s[l]->nb[1] * num_row;
                        uintptr_t row_end = (uintptr_t) buf + q11s[l]->nb[1] * (num_row + 1);
                        if (row_start < page_end && row_end > page_end && nodes[n] != nodes[n+1]) {
                            printf("row %d crosses pages from NUMA node %d and %d\n", num_row, nodes[n], nodes[n+1]);
                        }
                        if (row_end > page_end) {
                            break;
                        }
                    }
                    if (nodes[n] != current_node) {
                        printf("%d x %d, ", current_node_ctr, current_node);
                        current_node_ctr = 0;
                        current_node = nodes[n];
                        num_printed++;
                        if (num_printed % 8 == 0) {
                            printf("\n");
                        }
                    }
                    current_node_ctr++;
                }
                printf("%d x %d\n", current_node_ctr, current_node);
            }
        }
    }

//    ids->ne[1] = n_experts_used;
//    ids->nb[1] = ggml_row_size(ids->type, n_experts_used);

    // set tensor values
    for (int l = 0; l < n_layers; ++l) {
        ggml_set_f32(m11s[l], 1.0f);
    }
    ggml_set_f32(m2, 2.0f);
    ggml_set_f32(cur, 0.0f);

    // quantization
    int32_t nelements = sizex*sizey*n_experts;
    for (int l = 0; l < n_layers; ++l) {
        ggml_quantize_chunk(qtype, (const float *) m11s[l]->data, q11s[l]->data, 0, nelements/m11s[l]->ne[0], m11s[l]->ne[0], nullptr);
    }

    // compute graphs again after setting tensor values
    ggml_backend_graph_compute(backend, gf);
    ggml_backend_graph_compute(backend, gf31);

    TENSOR_DUMP(m11s[0]);
    TENSOR_DUMP(m2);
    TENSOR_DUMP(ggml_graph_node(gf, 0));

    printf("\n------ Test 2 - Matrix Mult via %s code\n", ggml_type_name(qtype));

    printf("n_threads=%i\n", benchmark_params.n_threads);

    const int dimx = sizex;
    const int dimy = sizey;
    const int dimz = sizez;
    long long int flops_per_dot_product = dimy + dimy;
    long long int flops_per_matrix = flops_per_dot_product * dimx * dimz; ;
    long long int flops_per_graph = flops_per_matrix * n_layers * n_experts; ;
    printf("Matrix Multiplication of (%i,%i,%i) x (%i,%i,%i) [%d layers] - about %6.2f gFLOPS\n\n", sizex, sizey, 1, sizex, sizez, 1, n_layers, 1.0f*flops_per_graph / 1000 / 1000 / 1000);


    // Let's use the F32 result from above as a reference for the quantized multiplication
    float sum_of_F32_reference = tensor_sum_elements(ggml_graph_node(gf, ggml_graph_n_nodes(gf) - 1));

    printf("Iteration;NThreads;NLayers; SizeX; SizeY; SizeZ; Required_FLOPS; Elapsed_u_Seconds; gigaFLOPS\n");
    printf("=====================================================================================\n");

    double  gflops_sum = 0;
    for (int i=0;i<benchmark_params.n_iterations ;i++) {

        long long int start = ggml_time_us();
        //printf("Running ggml_graph_compute\n");
        ggml_backend_graph_compute(backend, gf31);

        long long int stop = ggml_time_us();
        long long int usec = stop-start;
        double gflops = (double)(flops_per_graph)/usec/1000.0;
        gflops_sum += gflops;
        printf("%9i;%8i;%7i;%6i;%6i;%6i;%15lli;%18lli;%10.2f\n",
            i,
            benchmark_params.n_threads, n_layers,
            sizex, sizey, sizez, flops_per_matrix,
            usec,gflops);

#ifdef VERBOSE_DEBUGGING
        TENSOR_DUMP("res",gf31.nodes[0])
#endif

        // Check that the matrix multiplication result is in the right ballpark
        // We cannot use the exact value from the F32 multiplication because the quantizuation will be slightly different
        float sum_of_Q4_result = tensor_sum_elements(ggml_graph_node(gf31, ggml_graph_n_nodes(gf31) - 1));
        float delta = std::abs(sum_of_Q4_result - sum_of_F32_reference);
        float allowed_delta = (sum_of_F32_reference) / 1000 / 1000; //  Let's accept an epsilon of 10^-6

        if (delta > allowed_delta)  {
            printf("\nABORT - ERROR in Matrix Multiplication result - expected %6.2f, got %6.2f (delta %6.2f > allowed_delta %6.2f)\n",
                sum_of_F32_reference,
                sum_of_Q4_result,
                delta,
                allowed_delta
            );
            exit(0);
        }
    }
    printf("\n");
    printf("Average%86.2f\n",gflops_sum/((double)benchmark_params.n_iterations));
    printf("=============================================================================================\n");

    ggml_free(ctx);
    ggml_backend_buffer_free(buffer);
    ggml_backend_buffer_free(buffer2);
    ggml_backend_free(backend);
}
