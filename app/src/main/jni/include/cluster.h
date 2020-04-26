#ifndef min
#define min(x, y) ((x) < (y) ? (x) : (y))
#endif
#ifndef max
#define max(x, y) ((x) > (y) ? (x) : (y))
#endif
#ifdef WINDOWS
#include <windows.h>
#endif
#define CLUSTERVERSION "1.50"
double clusterdistance(int nrows, int ncolumns, double** data, int** mask,
                       double weight[], int n1, int n2, int index1[],
                       int index2[], char dist, char method, int transpose);
double** distancematrix(int ngenes, int ndata, double** data, int** mask,
                        double* weight, char dist, int transpose);
int getclustercentroids(int nclusters, int nrows, int ncolumns, double** data,
                        int** mask, int clusterid[], double** cdata,
                        int** cmask, int transpose, char method);
void getclustermedoids(int nclusters, int nelements, double** distance,
                       int clusterid[], int centroids[], double errors[]);
void kcluster(int nclusters, int ngenes, int ndata, double** data, int** mask,
              double weight[], int transpose, int npass, char method, char dist,
              int clusterid[], double* error, int* ifound);
void kmedoids(int nclusters, int nelements, double** distance, int npass,
              int clusterid[], double* error, int* ifound);
typedef struct {
  int left;
  int right;
  double distance;
} Node2;
Node2* treecluster(int nrows, int ncolumns, double** data, int** mask,
                  double weight[], int transpose, char dist, char method,
                  double** distmatrix);
void cuttree(int nelements, Node2* tree, int nclusters, int clusterid[]);
void somcluster(int nrows, int ncolumns, double** data, int** mask,
                const double weight[], int transpose, int nxnodes, int nynodes,
                double inittau, int niter, char dist, double*** celldata,
                int clusterid[][2]);
int pca(int m, int n, double** u, double** v, double* w);
void sort(int n, const double data[], int index[]);
double mean(int n, double x[]);
double median(int n, double x[]);
double* calculate_weights(int nrows, int ncolumns, double** data, int** mask,
                          double weights[], int transpose, char dist,
                          double cutoff, double exponent);
