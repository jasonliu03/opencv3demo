
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <limits.h>
#include <string.h>
#include "cluster.h"
#ifdef WINDOWS
#  include <windows.h>
#endif

#ifdef WINDOWS
int WINAPI
clusterdll_init (HANDLE h, DWORD reason, void* foo)
{
  return 1;
}
#endif
double mean(int n, double x[])
{ double result = 0.;
  int i;
  for (i = 0; i < n; i++) result += x[i];
  result /= n;
  return result;
}

double median (int n, double x[])
{ int i, j;
  int nr = n / 2;
  int nl = nr - 1;
  int even = 0;
  int lo = 0;
  int hi = n-1;

  if (n==2*nr) even = 1;
  if (n<3)
  { if (n<1) return 0.;
    if (n == 1) return x[0];
    return 0.5*(x[0]+x[1]);
  }
  do
  { int loop;
    int mid = (lo + hi)/2;
    double result = x[mid];
    double xlo = x[lo];
    double xhi = x[hi];
    if (xhi<xlo)
    { double temp = xlo;
      xlo = xhi;
      xhi = temp;
    }
    if (result>xhi) result = xhi;
    else if (result<xlo) result = xlo;
   
    i = lo;
    j = hi;
    do
    { while (x[i]<result) i++;
      while (x[j]>result) j--;
      loop = 0;
      if (i<j)
      { double temp = x[i];
        x[i] = x[j];
        x[j] = temp;
        i++;
        j--;
        if (i<=j) loop = 1;
      }
    } while (loop);

    if (even)
    { if (j==nl && i==nr)
        
        { int k;
          double xmax = x[0];
          double xmin = x[n-1];
          for (k = lo; k <= j; k++) xmax = max(xmax,x[k]);
          for (k = i; k <= hi; k++) xmin = min(xmin,x[k]);
          return 0.5*(xmin + xmax);
        }
      if (j<nl) lo = i;
      if (i>nr) hi = j;
      if (i==j)
      { if (i==nl) lo = nl;
        if (j==nr) hi = nr;
      }
    }
    else
    { if (j<nr) lo = i;
      if (i>nr) hi = j;
      
      if (i==j && i==nr) return result;
    }
  }
  while (lo<hi-1);

  if (even) return (0.5*(x[nl]+x[nr]));
  if (x[lo]>x[hi])
  { double temp = x[lo];
    x[lo] = x[hi];
    x[hi] = temp;
  }
  return x[nr];
}



static const double* sortdata = NULL;

static
int compare(const void* a, const void* b)

{ const int i1 = *(const int*)a;
  const int i2 = *(const int*)b;
  const double term1 = sortdata[i1];
  const double term2 = sortdata[i2];
  if (term1 < term2) return -1;
  if (term1 > term2) return +1;
  return 0;
}

void sort(int n, const double data[], int index[])

{ int i;
  sortdata = data;
  for (i = 0; i < n; i++) index[i] = i;
  qsort(index, n, sizeof(int), compare);
}



static double* getrank (int n, double data[])

{ int i;
  double* rank;
  int* index;
  rank = (double*)malloc(n*sizeof(double));
  if (!rank) return NULL;
  index = (int*)malloc(n*sizeof(int));
  if (!index)
  { free(rank);
    return NULL;
  }
  sort (n, data, index);

  for (i = 0; i < n; i++) rank[index[i]] = i;

  i = 0;
  while (i < n)
  { int m;
    double value = data[index[i]];
    int j = i + 1;
    while (j < n && data[index[j]] == value) j++;
    m = j - i;
    value = rank[index[i]] + (m-1)/2.;
    for (j = i; j < i + m; j++) rank[index[j]] = value;
    i += m;
  }
  free (index);
  return rank;
}

static int
makedatamask(int nrows, int ncols, double*** pdata, int*** pmask)
{ int i;
  double** data;
  int** mask;
  data = (double**)malloc(nrows*sizeof(double*));
  if(!data) return 0;
  mask = (int**)malloc(nrows*sizeof(int*));
  if(!mask)
  { free(data);
    return 0;
  }
  for (i = 0; i < nrows; i++)
  { data[i] = (double*)malloc(ncols*sizeof(double));
    if(!data[i]) break;
    mask[i] = (int*)malloc(ncols*sizeof(int));
    if(!mask[i])
    { free(data[i]);
      break;
    }
  }if (i==nrows) { *pdata = data;
    *pmask = mask;
    return 1;
  }
  *pdata = NULL;
  *pmask = NULL;
  nrows = i;
  for (i = 0; i < nrows; i++)
  { free(data[i]);
    free(mask[i]);
  }
  free(data);
  free(mask);
  return 0;
}

static void
freedatamask(int n, double** data, int** mask)
{ int i;
  for (i = 0; i < n; i++)
  { free(mask[i]);
    free(data[i]);
  }
  free(mask);
  free(data);
}

static
double find_closest_pair(int n, double** distmatrix, int* ip, int* jp)

{ int i, j;
  double temp;
  double distance = distmatrix[1][0];
  *ip = 1;
  *jp = 0;
  for (i = 1; i < n; i++)
  { for (j = 0; j < i; j++)
    { temp = distmatrix[i][j];
      if (temp<distance)
      { distance = temp;
        *ip = i;
        *jp = j;
      }
    }
  }
  return distance;
}

static int svd(int m, int n, double** u, double w[], double** vt)
{ int i, j, k, i1, k1, l1, its;
  double c,f,h,s,x,y,z;
  int l = 0;
  int ierr = 0;
  double g = 0.0;
  double scale = 0.0;
  double anorm = 0.0;
  double* rv1 = (double*)malloc(n*sizeof(double));
  if (!rv1) return -1;
  if (m >= n)
  { 
    for (i = 0; i < n; i++)
    { l = i + 1;
      rv1[i] = scale * g;
      g = 0.0;
      s = 0.0;
      scale = 0.0;
      for (k = i; k < m; k++) scale += fabs(u[k][i]);
      if (scale != 0.0)
      { for (k = i; k < m; k++)
        { u[k][i] /= scale;
          s += u[k][i]*u[k][i];
        }
        f = u[i][i];
        g = (f >= 0) ? -sqrt(s) : sqrt(s);
        h = f * g - s;
        u[i][i] = f - g;
        if (i < n-1)
        { for (j = l; j < n; j++)
          { s = 0.0;
            for (k = i; k < m; k++) s += u[k][i] * u[k][j];
            f = s / h;
            for (k = i; k < m; k++) u[k][j] += f * u[k][i];
          }
        }
        for (k = i; k < m; k++) u[k][i] *= scale;
      }
      w[i] = scale * g;
      g = 0.0;
      s = 0.0;
      scale = 0.0;
      if (i<n-1)
      { for (k = l; k < n; k++) scale += fabs(u[i][k]);
        if (scale != 0.0)
        { for (k = l; k < n; k++)
          { u[i][k] /= scale;
            s += u[i][k] * u[i][k];
          }
          f = u[i][l];
          g = (f >= 0) ? -sqrt(s) : sqrt(s);
          h = f * g - s;
          u[i][l] = f - g;
          for (k = l; k < n; k++) rv1[k] = u[i][k] / h;
          for (j = l; j < m; j++)
          { s = 0.0;
            for (k = l; k < n; k++) s += u[j][k] * u[i][k];
            for (k = l; k < n; k++) u[j][k] += s * rv1[k];
          }
          for (k = l; k < n; k++)  u[i][k] *= scale;
        }
      }
      anorm = max(anorm,fabs(w[i])+fabs(rv1[i]));
    }

    for (i = n-1; i>=0; i--)
    { if (i < n-1)
      { if (g != 0.0)
        { for (j = l; j < n; j++) vt[i][j] = (u[i][j] / u[i][l]) / g;

          for (j = l; j < n; j++)
          { s = 0.0;
            for (k = l; k < n; k++) s += u[i][k] * vt[j][k];
            for (k = l; k < n; k++) vt[j][k] += s * vt[i][k];
          }
        }
      }
      for (j = l; j < n; j++)
      { vt[j][i] = 0.0;
        vt[i][j] = 0.0;
      }
      vt[i][i] = 1.0;
      g = rv1[i];
      l = i;
    }

    for (i = n-1; i >= 0; i--)
    { l = i + 1;
      g = w[i];
      if (i!=n-1)
        for (j = l; j < n; j++) u[i][j] = 0.0;
      if (g!=0.0)
      { if (i!=n-1)
        { for (j = l; j < n; j++)
          { s = 0.0;
            for (k = l; k < m; k++) s += u[k][i] * u[k][j];

            f = (s / u[i][i]) / g;
            for (k = i; k < m; k++) u[k][j] += f * u[k][i];
          }
        }
        for (j = i; j < m; j++) u[j][i] /= g;
      }
      else
        for (j = i; j < m; j++) u[j][i] = 0.0;
      u[i][i] += 1.0;
    }

    for (k = n-1; k >= 0; k--)
    { k1 = k-1;
      its = 0;
      while(1)
      { for (l = k; l >= 0; l--)
        { l1 = l-1;
          if (fabs(rv1[l]) + anorm == anorm) break;
          if (fabs(w[l1]) + anorm == anorm)
          { c = 0.0;
            s = 1.0;
            for (i = l; i <= k; i++)
            { f = s * rv1[i];
              rv1[i] *= c;
              if (fabs(f) + anorm == anorm) break;
              g = w[i];
              h = sqrt(f*f+g*g);
              w[i] = h;
              c = g / h;
              s = -f / h;
              for (j = 0; j < m; j++)
              { y = u[j][l1];
                z = u[j][i];
                u[j][l1] = y * c + z * s;
                u[j][i] = -y * s + z * c;
              }
            }
            break;
          }
        }
        z = w[k];
        if (l==k) 
        { if (z < 0.0)
          { w[k] = -z;
            for (j = 0; j < n; j++) vt[k][j] = -vt[k][j];
          }
          break;
        }
        else if (its==30)
        { ierr = k;
          break;
        }
        else

        { its++;
          x = w[l];
          y = w[k1];
          g = rv1[k1];
          h = rv1[k];
          f = ((y - z) * (y + z) + (g - h) * (g + h)) / (2.0 * h * y);
          g = sqrt(f*f+1.0);
          f = ((x - z) * (x + z) + h * (y / (f + (f >= 0 ? g : -g)) - h)) / x;

          c = 1.0;
          s = 1.0;
          for (i1 = l; i1 <= k1; i1++)
          { i = i1 + 1;
            g = rv1[i];
            y = w[i];
            h = s * g;
            g = c * g;
            z = sqrt(f*f+h*h);
            rv1[i1] = z;
            c = f / z;
            s = h / z;
            f = x * c + g * s;
            g = -x * s + g * c;
            h = y * s;
            y = y * c;
            for (j = 0; j < n; j++)
            { x = vt[i1][j];
              z = vt[i][j];
              vt[i1][j] = x * c + z * s;
              vt[i][j] = -x * s + z * c;
            }
            z = sqrt(f*f+h*h);
            w[i1] = z;

            if (z!=0.0)
            { c = f / z;
              s = h / z;
            }
            f = c * g + s * y;
            x = -s * g + c * y;
            for (j = 0; j < m; j++)
            { y = u[j][i1];
              z = u[j][i];
              u[j][i1] = y * c + z * s;
              u[j][i] = -y * s + z * c;
            }
          }
          rv1[l] = 0.0;
          rv1[k] = f;
          w[k] = x;
        }
      }
    }
  }
  else 
  { 
    for (i = 0; i < m; i++)
    { l = i + 1;
      rv1[i] = scale * g;
      g = 0.0;
      s = 0.0;
      scale = 0.0;
      for (k = i; k < n; k++) scale += fabs(u[i][k]);
      if (scale != 0.0)
      { for (k = i; k < n; k++)
        { u[i][k] /= scale;
          s += u[i][k]*u[i][k];
        }
        f = u[i][i];
        g = (f >= 0) ? -sqrt(s) : sqrt(s);
        h = f * g - s;
        u[i][i] = f - g;
        if (i < m-1)
        { for (j = l; j < m; j++)
          { s = 0.0;
            for (k = i; k < n; k++) s += u[i][k] * u[j][k];
            f = s / h;
            for (k = i; k < n; k++) u[j][k] += f * u[i][k];
          }
        }
        for (k = i; k < n; k++) u[i][k] *= scale;
      }
      w[i] = scale * g;
      g = 0.0;
      s = 0.0;
      scale = 0.0;
      if (i<m-1)
      { for (k = l; k < m; k++) scale += fabs(u[k][i]);
        if (scale != 0.0)
        { for (k = l; k < m; k++)
          { u[k][i] /= scale;
            s += u[k][i] * u[k][i];
          }
          f = u[l][i];
          g = (f >= 0) ? -sqrt(s) : sqrt(s);
          h = f * g - s;
          u[l][i] = f - g;
          for (k = l; k < m; k++) rv1[k] = u[k][i] / h;
          for (j = l; j < n; j++)
          { s = 0.0;
            for (k = l; k < m; k++) s += u[k][j] * u[k][i];
            for (k = l; k < m; k++) u[k][j] += s * rv1[k];
          }
          for (k = l; k < m; k++)  u[k][i] *= scale;
        }
      }
      anorm = max(anorm,fabs(w[i])+fabs(rv1[i]));
    }

    for (i = m-1; i>=0; i--)
    { if (i < m-1)
      { if (g != 0.0)
        { for (j = l; j < m; j++) vt[j][i] = (u[j][i] / u[l][i]) / g;

          for (j = l; j < m; j++)
          { s = 0.0;
            for (k = l; k < m; k++) s += u[k][i] * vt[k][j];
            for (k = l; k < m; k++) vt[k][j] += s * vt[k][i];
          }
        }
      }
      for (j = l; j < m; j++)
      { vt[i][j] = 0.0;
        vt[j][i] = 0.0;
      }
      vt[i][i] = 1.0;
      g = rv1[i];
      l = i;
    }

    for (i = m-1; i >= 0; i--)
    { l = i + 1;
      g = w[i];
      if (i!=m-1)
        for (j = l; j < m; j++) u[j][i] = 0.0;
      if (g!=0.0)
      { if (i!=m-1)
        { for (j = l; j < m; j++)
          { s = 0.0;
            for (k = l; k < n; k++) s += u[i][k] * u[j][k];

            f = (s / u[i][i]) / g;
            for (k = i; k < n; k++) u[j][k] += f * u[i][k];
          }
        }
        for (j = i; j < n; j++) u[i][j] /= g;
      }
      else
        for (j = i; j < n; j++) u[i][j] = 0.0;
      u[i][i] += 1.0;
    }
    for (k = m-1; k >= 0; k--)
    { k1 = k-1;
      its = 0;
      while(1)
      { for (l = k; l >= 0; l--)
        { l1 = l-1;
          if (fabs(rv1[l]) + anorm == anorm) break;
          if (fabs(w[l1]) + anorm == anorm)
          { c = 0.0;
            s = 1.0;
            for (i = l; i <= k; i++)
            { f = s * rv1[i];
              rv1[i] *= c;
              if (fabs(f) + anorm == anorm) break;
              g = w[i];
              h = sqrt(f*f+g*g);
              w[i] = h;
              c = g / h;
              s = -f / h;
              for (j = 0; j < n; j++)
              { y = u[l1][j];
                z = u[i][j];
                u[l1][j] = y * c + z * s;
                u[i][j] = -y * s + z * c;
              }
            }
            break;
          }
        }
        z = w[k];
        if (l==k) 
        { if (z < 0.0)
          { w[k] = -z;
            for (j = 0; j < m; j++) vt[j][k] = -vt[j][k];
          }
          break;
        }
        else if (its==30)
        { ierr = k;
          break;
        }
        else
        { its++;
          x = w[l];
          y = w[k1];
          g = rv1[k1];
          h = rv1[k];
          f = ((y - z) * (y + z) + (g - h) * (g + h)) / (2.0 * h * y);
          g = sqrt(f*f+1.0);
          f = ((x - z) * (x + z) + h * (y / (f + (f >= 0 ? g : -g)) - h)) / x;
       
          c = 1.0;
          s = 1.0;
          for (i1 = l; i1 <= k1; i1++)
          { i = i1 + 1;
            g = rv1[i];
            y = w[i];
            h = s * g;
            g = c * g;
            z = sqrt(f*f+h*h);
            rv1[i1] = z;
            c = f / z;
            s = h / z;
            f = x * c + g * s;
            g = -x * s + g * c;
            h = y * s;
            y = y * c;
            for (j = 0; j < m; j++)
            { x = vt[j][i1];
              z = vt[j][i];
              vt[j][i1] = x * c + z * s;
              vt[j][i] = -x * s + z * c;
            }
            z = sqrt(f*f+h*h);
            w[i1] = z;

            if (z!=0.0)
            { c = f / z;
              s = h / z;
            }
            f = c * g + s * y;
            x = -s * g + c * y;
            for (j = 0; j < n; j++)
            { y = u[i1][j];
              z = u[i][j];
              u[i1][j] = y * c + z * s;
              u[i][j] = -y * s + z * c;
            }
          }
          rv1[l] = 0.0;
          rv1[k] = f;
          w[k] = x;
        }
      }
    }
  }
  free(rv1);
  return ierr;
}



int pca(int nrows, int ncolumns, double** u, double** v, double* w)

{
    int i;
    int j;
    int error;
    int* index = (int*)malloc(ncolumns*sizeof(int));
    double* temp = (double*)malloc(ncolumns*sizeof(double));
    if (!index || !temp)
    {   if (index) free(index);
        if (temp) free(temp);
        return -1;
    }
    error = svd(nrows, ncolumns, u, w, v);
    if (error==0)
    {
        if (nrows >= ncolumns)
        {   for (j = 0; j < ncolumns; j++)
            {   const double s = w[j];
                for (i = 0; i < nrows; i++) u[i][j] *= s;
            }
            sort(ncolumns, w, index);
            for (i = 0; i < ncolumns/2; i++)
            {   j = index[i];
                index[i] = index[ncolumns-1-i];
                index[ncolumns-1-i] = j;
            }
            for (i = 0; i < nrows; i++)
            {   for (j = 0; j < ncolumns; j++) temp[j] = u[i][index[j]];
                for (j = 0; j < ncolumns; j++) u[i][j] = temp[j];
            }
            for (i = 0; i < ncolumns; i++)
            {   for (j = 0; j < ncolumns; j++) temp[j] = v[index[j]][i];
                for (j = 0; j < ncolumns; j++) v[j][i] = temp[j];
            }
            for (i = 0; i < ncolumns; i++) temp[i] = w[index[i]];
            for (i = 0; i < ncolumns; i++) w[i] = temp[i];
        }
        else 
        {   for (j = 0; j < nrows; j++)
            {   const double s = w[j];
                for (i = 0; i < nrows; i++) v[i][j] *= s;
            }
            sort(nrows, w, index);
            for (i = 0; i < nrows/2; i++)
            {   j = index[i];
                index[i] = index[nrows-1-i];
                index[nrows-1-i] = j;
            }
            for (j = 0; j < ncolumns; j++)
            {   for (i = 0; i < nrows; i++) temp[i] = u[index[i]][j];
                for (i = 0; i < nrows; i++) u[i][j] = temp[i];
            }
            for (j = 0; j < nrows; j++)
            {   for (i = 0; i < nrows; i++) temp[i] = v[j][index[i]];
                for (i = 0; i < nrows; i++) v[j][i] = temp[i];
            }
            for (i = 0; i < nrows; i++) temp[i] = w[index[i]];
            for (i = 0; i < nrows; i++) w[i] = temp[i];
        }
    }
    free(index);
    free(temp);
    return error;
}



static
double euclid (int n, double** data1, double** data2, int** mask1, int** mask2,
  const double weight[], int index1, int index2, int transpose)
 

{ double result = 0.;
  double tweight = 0;
  int i;
  if (transpose==0) 
  { for (i = 0; i < n; i++)
    { if (mask1[index1][i] && mask2[index2][i])
      { double term = data1[index1][i] - data2[index2][i];
        result += weight[i]*term*term;
        tweight += weight[i];
      }
    }
  }
  else
  { for (i = 0; i < n; i++)
    { if (mask1[i][index1] && mask2[i][index2])
      { double term = data1[i][index1] - data2[i][index2];
        result += weight[i]*term*term;
        tweight += weight[i];
      }
    }
  }
  if (!tweight) return 0; 
  result /= tweight;
  return result;
}



static
double cityblock (int n, double** data1, double** data2, int** mask1,
  int** mask2, const double weight[], int index1, int index2, int transpose)


{ double result = 0.;
  double tweight = 0;
  int i;
  if (transpose==0)
  { for (i = 0; i < n; i++)
    { if (mask1[index1][i] && mask2[index2][i])
      { double term = data1[index1][i] - data2[index2][i];
        result = result + weight[i]*fabs(term);
        tweight += weight[i];
      }
    }
  }
  else
  { for (i = 0; i < n; i++)
    { if (mask1[i][index1] && mask2[i][index2])
      { double term = data1[i][index1] - data2[i][index2];
        result = result + weight[i]*fabs(term);
        tweight += weight[i];
      }
    }
  }
  if (!tweight) return 0; 
  result /= tweight;
  return result;
}


static
double correlation (int n, double** data1, double** data2, int** mask1,
  int** mask2, const double weight[], int index1, int index2, int transpose)

{ double result = 0.;
  double sum1 = 0.;
  double sum2 = 0.;
  double denom1 = 0.;
  double denom2 = 0.;
  double tweight = 0.;
  if (transpose==0) 
  { int i;
    for (i = 0; i < n; i++)
    { if (mask1[index1][i] && mask2[index2][i])
      { double term1 = data1[index1][i];
        double term2 = data2[index2][i];
        double w = weight[i];
        sum1 += w*term1;
        sum2 += w*term2;
        result += w*term1*term2;
        denom1 += w*term1*term1;
        denom2 += w*term2*term2;
        tweight += w;
      }
    }
  }
  else
  { int i;
    for (i = 0; i < n; i++)
    { if (mask1[i][index1] && mask2[i][index2])
      { double term1 = data1[i][index1];
        double term2 = data2[i][index2];
        double w = weight[i];
        sum1 += w*term1;
        sum2 += w*term2;
        result += w*term1*term2;
        denom1 += w*term1*term1;
        denom2 += w*term2*term2;
        tweight += w;
      }
    }
  }
  if (!tweight) return 0;
  result -= sum1 * sum2 / tweight;
  denom1 -= sum1 * sum1 / tweight;
  denom2 -= sum2 * sum2 / tweight;
  if (denom1 <= 0) return 1;
  if (denom2 <= 0) return 1; 
  result = result / sqrt(denom1*denom2);
  result = 1. - result;
  return result;
}


static
double acorrelation (int n, double** data1, double** data2, int** mask1,
  int** mask2, const double weight[], int index1, int index2, int transpose)

{ double result = 0.;
  double sum1 = 0.;
  double sum2 = 0.;
  double denom1 = 0.;
  double denom2 = 0.;
  double tweight = 0.;
  if (transpose==0)
  { int i;
    for (i = 0; i < n; i++)
    { if (mask1[index1][i] && mask2[index2][i])
      { double term1 = data1[index1][i];
        double term2 = data2[index2][i];
        double w = weight[i];
        sum1 += w*term1;
        sum2 += w*term2;
        result += w*term1*term2;
        denom1 += w*term1*term1;
        denom2 += w*term2*term2;
        tweight += w;
      }
    }
  }
  else
  { int i;
    for (i = 0; i < n; i++)
    { if (mask1[i][index1] && mask2[i][index2])
      { double term1 = data1[i][index1];
        double term2 = data2[i][index2];
        double w = weight[i];
        sum1 += w*term1;
        sum2 += w*term2;
        result += w*term1*term2;
        denom1 += w*term1*term1;
        denom2 += w*term2*term2;
        tweight += w;
      }
    }
  }
  if (!tweight) return 0; 
  result -= sum1 * sum2 / tweight;
  denom1 -= sum1 * sum1 / tweight;
  denom2 -= sum2 * sum2 / tweight;
  if (denom1 <= 0) return 1;
  if (denom2 <= 0) return 1; 
  result = fabs(result) / sqrt(denom1*denom2);
  result = 1. - result;
  return result;
}



static
double ucorrelation (int n, double** data1, double** data2, int** mask1,
  int** mask2, const double weight[], int index1, int index2, int transpose)

{ double result = 0.;
  double denom1 = 0.;
  double denom2 = 0.;
  int flag = 0;

  if (transpose==0)
  { int i;
    for (i = 0; i < n; i++)
    { if (mask1[index1][i] && mask2[index2][i])
      { double term1 = data1[index1][i];
        double term2 = data2[index2][i];
        double w = weight[i];
        result += w*term1*term2;
        denom1 += w*term1*term1;
        denom2 += w*term2*term2;
        flag = 1;
      }
    }
  }
  else
  { int i;
    for (i = 0; i < n; i++)
    { if (mask1[i][index1] && mask2[i][index2])
      { double term1 = data1[i][index1];
        double term2 = data2[i][index2];
        double w = weight[i];
        result += w*term1*term2;
        denom1 += w*term1*term1;
        denom2 += w*term2*term2;
        flag = 1;
      }
    }
  }
  if (!flag) return 0.;
  if (denom1==0.) return 1.;
  if (denom2==0.) return 1.;
  result = result / sqrt(denom1*denom2);
  result = 1. - result;
  return result;
}


static
double uacorrelation (int n, double** data1, double** data2, int** mask1,
  int** mask2, const double weight[], int index1, int index2, int transpose)

{ double result = 0.;
  double denom1 = 0.;
  double denom2 = 0.;
  int flag = 0;
 
  if (transpose==0) 
  { int i;
    for (i = 0; i < n; i++)
    { if (mask1[index1][i] && mask2[index2][i])
      { double term1 = data1[index1][i];
        double term2 = data2[index2][i];
        double w = weight[i];
        result += w*term1*term2;
        denom1 += w*term1*term1;
        denom2 += w*term2*term2;
        flag = 1;
      }
    }
  }
  else
  { int i;
    for (i = 0; i < n; i++)
    { if (mask1[i][index1] && mask2[i][index2])
      { double term1 = data1[i][index1];
        double term2 = data2[i][index2];
        double w = weight[i];
        result += w*term1*term2;
        denom1 += w*term1*term1;
        denom2 += w*term2*term2;
        flag = 1;
      }
    }
  }
  if (!flag) return 0.;
  if (denom1==0.) return 1.;
  if (denom2==0.) return 1.;
  result = fabs(result) / sqrt(denom1*denom2);
  result = 1. - result;
  return result;
}


static
double spearman (int n, double** data1, double** data2, int** mask1,
  int** mask2, const double weight[], int index1, int index2, int transpose)

{ int i;
  int m = 0;
  double* rank1;
  double* rank2;
  double result = 0.;
  double denom1 = 0.;
  double denom2 = 0.;
  double avgrank;
  double* tdata1;
  double* tdata2;
  tdata1 = (double*)malloc(n*sizeof(double));
  if(!tdata1) return 0.0; 
  tdata2 = (double*)malloc(n*sizeof(double));
  if(!tdata2) 
  { free(tdata1);
    return 0.0;
  }
  if (transpose==0)
  { for (i = 0; i < n; i++)
    { if (mask1[index1][i] && mask2[index2][i])
      { tdata1[m] = data1[index1][i];
        tdata2[m] = data2[index2][i];
        m++;
      }
    }
  }
  else
  { for (i = 0; i < n; i++)
    { if (mask1[i][index1] && mask2[i][index2])
      { tdata1[m] = data1[i][index1];
        tdata2[m] = data2[i][index2];
        m++;
      }
    }
  }
  if (m==0)
  { free(tdata1);
    free(tdata2);
    return 0;
  }
  rank1 = getrank(m, tdata1);
  free(tdata1);
  if(!rank1)
  { free(tdata2);
    return 0.0; 
  }
  rank2 = getrank(m, tdata2);
  free(tdata2);
  if(!rank2) 
  { free(rank1);
    return 0.0;
  }
  avgrank = 0.5*(m-1); 
  for (i = 0; i < m; i++)
  { const double value1 = rank1[i];
    const double value2 = rank2[i];
    result += value1 * value2;
    denom1 += value1 * value1;
    denom2 += value2 * value2;
  }

  free(rank1);
  free(rank2);
  result /= m;
  denom1 /= m;
  denom2 /= m;
  result -= avgrank * avgrank;
  denom1 -= avgrank * avgrank;
  denom2 -= avgrank * avgrank;
  if (denom1 <= 0) return 1; 
  if (denom2 <= 0) return 1; 
  result = result / sqrt(denom1*denom2);
  result = 1. - result;
  return result;
}


static
double kendall (int n, double** data1, double** data2, int** mask1, int** mask2,
  const double weight[], int index1, int index2, int transpose)

{ int con = 0;
  int dis = 0;
  int exx = 0;
  int exy = 0;
  int flag = 0;

  double denomx;
  double denomy;
  double tau;
  int i, j;
  if (transpose==0)
  { for (i = 0; i < n; i++)
    { if (mask1[index1][i] && mask2[index2][i])
      { for (j = 0; j < i; j++)
        { if (mask1[index1][j] && mask2[index2][j])
          { double x1 = data1[index1][i];
            double x2 = data1[index1][j];
            double y1 = data2[index2][i];
            double y2 = data2[index2][j];
            if (x1 < x2 && y1 < y2) con++;
            if (x1 > x2 && y1 > y2) con++;
            if (x1 < x2 && y1 > y2) dis++;
            if (x1 > x2 && y1 < y2) dis++;
            if (x1 == x2 && y1 != y2) exx++;
            if (x1 != x2 && y1 == y2) exy++;
            flag = 1;
          }
        }
      }
    }
  }
  else
  { for (i = 0; i < n; i++)
    { if (mask1[i][index1] && mask2[i][index2])
      { for (j = 0; j < i; j++)
        { if (mask1[j][index1] && mask2[j][index2])
          { double x1 = data1[i][index1];
            double x2 = data1[j][index1];
            double y1 = data2[i][index2];
            double y2 = data2[j][index2];
            if (x1 < x2 && y1 < y2) con++;
            if (x1 > x2 && y1 > y2) con++;
            if (x1 < x2 && y1 > y2) dis++;
            if (x1 > x2 && y1 < y2) dis++;
            if (x1 == x2 && y1 != y2) exx++;
            if (x1 != x2 && y1 == y2) exy++;
            flag = 1;
          }
        }
      }
    }
  }
  if (!flag) return 0.;
  denomx = con + dis + exx;
  denomy = con + dis + exy;
  if (denomx==0) return 1;
  if (denomy==0) return 1;
  tau = (con-dis)/sqrt(denomx*denomy);
  return 1.-tau;
}


static double(*setmetric(char dist)) 
  (int, double**, double**, int**, int**, const double[], int, int, int)
{ switch(dist)
  { case 'e': return &euclid;
    case 'b': return &cityblock;
    case 'c': return &correlation;
    case 'a': return &acorrelation;
    case 'u': return &ucorrelation;
    case 'x': return &uacorrelation;
    case 's': return &spearman;
    case 'k': return &kendall;
    default: return &euclid;
  }
  return NULL; 
}

static double uniform(void)

{ int z;
  static const int m1 = 2147483563;
  static const int m2 = 2147483399;
  const double scale = 1.0/m1;

  static int s1 = 0;
  static int s2 = 0;

  if (s1==0 || s2==0) /* initialize */
  { unsigned int initseed = (unsigned int) time(0);
    srand(initseed);
    s1 = rand();
    s2 = rand();
  }

  do
  { int k;
    k = s1/53668;
    s1 = 40014*(s1-k*53668)-k*12211;
    if (s1 < 0) s1+=m1;
    k = s2/52774;
    s2 = 40692*(s2-k*52774)-k*3791;
    if(s2 < 0) s2+=m2;
    z = s1-s2;
    if(z < 1) z+=(m1-1);
  } while (z==m1); 

  return z*scale;
}

static int binomial(int n, double p)

{ const double q = 1 - p;
  if (n*p < 30.0) 
  { const double s = p/q;
    const double a = (n+1)*s;
    double r = exp(n*log(q)); 
    int x = 0;
    double u = uniform();
    while(1)
    { if (u < r) return x;
      u-=r;
      x++;
      r *= (a/x)-s;
    }
  }
  else 
  { 
    const double fm = n*p + p;
    const int m = (int) fm;
    const double p1 = floor(2.195*sqrt(n*p*q) -4.6*q) + 0.5;
    const double xm = m + 0.5;
    const double xl = xm - p1;
    const double xr = xm + p1;
    const double c = 0.134 + 20.5/(15.3+m);
    const double a = (fm-xl)/(fm-xl*p);
    const double b = (xr-fm)/(xr*q);
    const double lambdal = a*(1.0+0.5*a);
    const double lambdar = b*(1.0+0.5*b);
    const double p2 = p1*(1+2*c);
    const double p3 = p2 + c/lambdal;
    const double p4 = p3 + c/lambdar;
    while (1)
    {
      int y;
      int k;
      double u = uniform();
      double v = uniform();
      u *= p4;
      if (u <= p1) return (int)(xm-p1*v+u);
      
      if (u > p2)
      { 
        if (u > p3)
        { 
          y = (int)(xr-log(v)/lambdar);
          if (y > n) continue;
          
          v = v*(u-p3)*lambdar;
        }
        else
        { y = (int)(xl+log(v)/lambdal);
          if (y < 0) continue;
          
          v = v*(u-p2)*lambdal;
        }
      }
      else
      { const double x = xl + (u-p1)/c;
        v = v*c + 1.0 - fabs(m-x+0.5)/p1;
        if (v > 1) continue;
        
        y = (int)x;
      }
     
      k = abs(y-m);
      if (k > 20 && k < 0.5*n*p*q-1.0)
      { 
        double rho = (k/(n*p*q))*((k*(k/3.0 + 0.625) + 0.1666666666666)/(n*p*q)+0.5);
        double t = -k*k/(2*n*p*q);
        double A = log(v);
        if (A < t-rho) return y;
        else if (A > t+rho) continue;
        else
        { 
          double x1 = y+1;
          double f1 = m+1;
          double z = n+1-m;
          double w = n-y+1;
          double x2 = x1*x1;
          double f2 = f1*f1;
          double z2 = z*z;
          double w2 = w*w;
          if (A > xm * log(f1/x1) + (n-m+0.5)*log(z/w)
                + (y-m)*log(w*p/(x1*q))
                + (13860.-(462.-(132.-(99.-140./f2)/f2)/f2)/f2)/f1/166320.
                + (13860.-(462.-(132.-(99.-140./z2)/z2)/z2)/z2)/z/166320.
                + (13860.-(462.-(132.-(99.-140./x2)/x2)/x2)/x2)/x1/166320.
                + (13860.-(462.-(132.-(99.-140./w2)/w2)/w2)/w2)/w/166320.)
             continue;
          return y;
        }
      }
      else
      { 
        int i;
        const double s = p/q;
        const double aa = s*(n+1);
        double f = 1.0;
        for (i = m; i < y; f *= (aa/(++i)-s));
        for (i = y; i < m; f /= (aa/(++i)-s));
        if (v > f) continue;
        return y;
      }
    }
  }

  return -1;
}

static void randomassign (int nclusters, int nelements, int clusterid[])

{ int i, j;
  int k = 0;
  double p;
  int n = nelements-nclusters;
  
  for (i = 0; i < nclusters-1; i++)
  { p = 1.0/(nclusters-i);
    j = binomial(n, p);
    n -= j;
    j += k+1; 
    for ( ; k < j; k++) clusterid[k] = i;
  }

  for ( ; k < nelements; k++) clusterid[k] = i;


  for (i = 0; i < nelements; i++)
  { j = (int) (i + (nelements-i)*uniform());
    k = clusterid[j];
    clusterid[j] = clusterid[i];
    clusterid[i] = k;
  }

  return;
}



static void getclustermeans(int nclusters, int nrows, int ncolumns,
  double** data, int** mask, int clusterid[], double** cdata, int** cmask,
  int transpose)

{ int i, j, k;
  if (transpose==0)
  { for (i = 0; i < nclusters; i++)
    { for (j = 0; j < ncolumns; j++)
      { cmask[i][j] = 0;
        cdata[i][j] = 0.;
      }
    }
    for (k = 0; k < nrows; k++)
    { i = clusterid[k];
      for (j = 0; j < ncolumns; j++)
      { if (mask[k][j] != 0)
        { cdata[i][j]+=data[k][j];
          cmask[i][j]++;
        }
      }
    }
    for (i = 0; i < nclusters; i++)
    { for (j = 0; j < ncolumns; j++)
      { if (cmask[i][j]>0)
        { cdata[i][j] /= cmask[i][j];
          cmask[i][j] = 1;
        }
      }
    }
  }
  else
  { for (i = 0; i < nrows; i++)
    { for (j = 0; j < nclusters; j++)
      { cdata[i][j] = 0.;
        cmask[i][j] = 0;
      }
    }
    for (k = 0; k < ncolumns; k++)
    { i = clusterid[k];
      for (j = 0; j < nrows; j++)
      { if (mask[j][k] != 0)
        { cdata[j][i]+=data[j][k];
          cmask[j][i]++;
        }
      }
    }
    for (i = 0; i < nrows; i++)
    { for (j = 0; j < nclusters; j++)
      { if (cmask[i][j]>0)
        { cdata[i][j] /= cmask[i][j];
          cmask[i][j] = 1;
        }
      }
    }
  }
}



static void
getclustermedians(int nclusters, int nrows, int ncolumns,
  double** data, int** mask, int clusterid[], double** cdata, int** cmask,
  int transpose, double cache[])

{ int i, j, k;
  if (transpose==0)
  { for (i = 0; i < nclusters; i++)
    { for (j = 0; j < ncolumns; j++)
      { int count = 0;
        for (k = 0; k < nrows; k++)
        { if (i==clusterid[k] && mask[k][j])
          { cache[count] = data[k][j];
            count++;
          }
        }
        if (count>0)
        { cdata[i][j] = median(count,cache);
          cmask[i][j] = 1;
        }
        else
        { cdata[i][j] = 0.;
          cmask[i][j] = 0;
        }
      }
    }
  }
  else
  { for (i = 0; i < nclusters; i++)
    { for (j = 0; j < nrows; j++)
      { int count = 0;
        for (k = 0; k < ncolumns; k++)
        { if (i==clusterid[k] && mask[j][k])
          { cache[count] = data[j][k];
            count++;
          }
        }
        if (count>0)
        { cdata[j][i] = median(count,cache);
          cmask[j][i] = 1;
        }
        else
        { cdata[j][i] = 0.;
          cmask[j][i] = 0;
        }
      }
    }
  }
}
 


int getclustercentroids(int nclusters, int nrows, int ncolumns,
  double** data, int** mask, int clusterid[], double** cdata, int** cmask,
  int transpose, char method)

{ switch(method)
  { case 'm':
    { const int nelements = (transpose==0) ? nrows : ncolumns;
      double* cache = (double*)malloc(nelements*sizeof(double));
      if (!cache) return 0;
      getclustermedians(nclusters, nrows, ncolumns, data, mask, clusterid,
                        cdata, cmask, transpose, cache);
      free(cache);
      return 1;
    }
    case 'a':
    { getclustermeans(nclusters, nrows, ncolumns, data, mask, clusterid,
                      cdata, cmask, transpose);
      return 1;
    }
  }
  return 0;
}


void getclustermedoids(int nclusters, int nelements, double** distance,
  int clusterid[], int centroids[], double errors[])

{ int i, j, k;
  for (j = 0; j < nclusters; j++) errors[j] = DBL_MAX;
  for (i = 0; i < nelements; i++)
  { double d = 0.0;
    j = clusterid[i];
    for (k = 0; k < nelements; k++)
    { if (i==k || clusterid[k]!=j) continue;
      d += (i < k ? distance[k][i] : distance[i][k]);
      if (d > errors[j]) break;
    }
    if (d < errors[j])
    { errors[j] = d;
      centroids[j] = i;
    }
  }
}



static int
kmeans(int nclusters, int nrows, int ncolumns, double** data, int** mask,
  double weight[], int transpose, int npass, char dist,
  double** cdata, int** cmask, int clusterid[], double* error,
  int tclusterid[], int counts[], int mapping[])
{ int i, j, k;
  const int nelements = (transpose==0) ? nrows : ncolumns;
  const int ndata = (transpose==0) ? ncolumns : nrows;
  int ifound = 1;
  int ipass = 0;

  double (*metric)
    (int, double**, double**, int**, int**, const double[], int, int, int) =
       setmetric(dist);

  
  int* saved = (int*)malloc(nelements*sizeof(int));
  if (saved==NULL) return -1;

  *error = DBL_MAX;

  do
  { double total = DBL_MAX;
    int counter = 0;
    int period = 10;

   
    if (npass!=0) randomassign (nclusters, nelements, tclusterid);

    for (i = 0; i < nclusters; i++) counts[i] = 0;
    for (i = 0; i < nelements; i++) counts[tclusterid[i]]++;

   
    while(1)
    { double previous = total;
      total = 0.0;

      if (counter % period == 0) /* Save the current cluster assignments */
      { for (i = 0; i < nelements; i++) saved[i] = tclusterid[i];
        if (period < INT_MAX / 2) period *= 2;
      }
      counter++;

     
      getclustermeans(nclusters, nrows, ncolumns, data, mask, tclusterid,
                      cdata, cmask, transpose);

      for (i = 0; i < nelements; i++)
     
      { double distance;
        k = tclusterid[i];
        if (counts[k]==1) continue;
       
        distance = metric(ndata,data,cdata,mask,cmask,weight,i,k,transpose);
        for (j = 0; j < nclusters; j++)
        { double tdistance;
          if (j==k) continue;
          tdistance = metric(ndata,data,cdata,mask,cmask,weight,i,j,transpose);
          if (tdistance < distance)
          { distance = tdistance;
            counts[tclusterid[i]]--;
            tclusterid[i] = j;
            counts[j]++;
          }
        }
        total += distance;
      }
      if (total>=previous) break;
     
      for (i = 0; i < nelements; i++)
        if (saved[i]!=tclusterid[i]) break;
      if (i==nelements)
        break; 
    }

    if (npass<=1) 
    { *error = total;
      break;
    }

    for (i = 0; i < nclusters; i++) mapping[i] = -1;
    for (i = 0; i < nelements; i++)
    { j = tclusterid[i];
      k = clusterid[i];
      if (mapping[k] == -1) mapping[k] = j;
      else if (mapping[k] != j)
      { if (total < *error)
        { ifound = 1;
          *error = total;
          for (j = 0; j < nelements; j++) clusterid[j] = tclusterid[j];
        }
        break;
      }
    }
    if (i==nelements) ifound++; 
  } while (++ipass < npass);

  free(saved);
  return ifound;
}

static int
kmedians(int nclusters, int nrows, int ncolumns, double** data, int** mask,
  double weight[], int transpose, int npass, char dist,
  double** cdata, int** cmask, int clusterid[], double* error,
  int tclusterid[], int counts[], int mapping[], double cache[])
{ int i, j, k;
  const int nelements = (transpose==0) ? nrows : ncolumns;
  const int ndata = (transpose==0) ? ncolumns : nrows;
  int ifound = 1;
  int ipass = 0;
 
  double (*metric)
    (int, double**, double**, int**, int**, const double[], int, int, int) =
       setmetric(dist);

 
  int* saved = (int*)malloc(nelements*sizeof(int));
  if (saved==NULL) return -1;

  *error = DBL_MAX;

  do
  { double total = DBL_MAX;
    int counter = 0;
    int period = 10;

    
    if (npass!=0) randomassign (nclusters, nelements, tclusterid);

    for (i = 0; i < nclusters; i++) counts[i]=0;
    for (i = 0; i < nelements; i++) counts[tclusterid[i]]++;

    while(1)
    { double previous = total;
      total = 0.0;

      if (counter % period == 0) 
      { for (i = 0; i < nelements; i++) saved[i] = tclusterid[i];
        if (period < INT_MAX / 2) period *= 2;
      }
      counter++;

   
      getclustermedians(nclusters, nrows, ncolumns, data, mask, tclusterid,
                        cdata, cmask, transpose, cache);

      for (i = 0; i < nelements; i++)
   
      { double distance;
        k = tclusterid[i];
        if (counts[k]==1) continue;

        distance = metric(ndata,data,cdata,mask,cmask,weight,i,k,transpose);
        for (j = 0; j < nclusters; j++)
        { double tdistance;
          if (j==k) continue;
          tdistance = metric(ndata,data,cdata,mask,cmask,weight,i,j,transpose);
          if (tdistance < distance)
          { distance = tdistance;
            counts[tclusterid[i]]--;
            tclusterid[i] = j;
            counts[j]++;
          }
        }
        total += distance;
      }
      if (total>=previous) break;

      for (i = 0; i < nelements; i++)
        if (saved[i]!=tclusterid[i]) break;
      if (i==nelements)
        break; 
    }

    if (npass<=1) 
    { *error = total;
      break;
    }

    for (i = 0; i < nclusters; i++) mapping[i] = -1;
    for (i = 0; i < nelements; i++)
    { j = tclusterid[i];
      k = clusterid[i];
      if (mapping[k] == -1) mapping[k] = j;
      else if (mapping[k] != j)
      { if (total < *error)
        { ifound = 1;
          *error = total;
          for (j = 0; j < nelements; j++) clusterid[j] = tclusterid[j];
        }
        break;
      }
    }
    if (i==nelements) ifound++; 
  } while (++ipass < npass);

  free(saved);
  return ifound;
}

void kcluster (int nclusters, int nrows, int ncolumns,
  double** data, int** mask, double weight[], int transpose,
  int npass, char method, char dist,
  int clusterid[], double* error, int* ifound)

{ const int nelements = (transpose==0) ? nrows : ncolumns;
  const int ndata = (transpose==0) ? ncolumns : nrows;

  int i;
  int ok;
  int* tclusterid;
  int* mapping = NULL;
  double** cdata;
  int** cmask;
  int* counts;

  if (nelements < nclusters)
  { *ifound = 0;
    return;
  }


  *ifound = -1;


  counts = (int*)malloc(nclusters*sizeof(int));
  if(!counts) return;
  if (npass<=1) tclusterid = clusterid;
  else
  { tclusterid = (int*)malloc(nelements*sizeof(int));
    if (!tclusterid)
    { free(counts);
      return;
    }
    mapping = (int*)malloc(nclusters*sizeof(int));
    if (!mapping)
    { free(counts);
      free(tclusterid);
      return;
    }
    for (i = 0; i < nelements; i++) clusterid[i] = 0;
  }


  if (transpose==0) ok = makedatamask(nclusters, ndata, &cdata, &cmask);
  else ok = makedatamask(ndata, nclusters, &cdata, &cmask);
  if(!ok)
  { free(counts);
    if(npass>1)
    { free(tclusterid);
      free(mapping);
      return;
    }
  }
  
  if (method=='m')
  { double* cache = (double*)malloc(nelements*sizeof(double));
    if(cache)
    { *ifound = kmedians(nclusters, nrows, ncolumns, data, mask, weight,
                         transpose, npass, dist, cdata, cmask, clusterid, error,
                         tclusterid, counts, mapping, cache);
      free(cache);
    }
  }
  else
    *ifound = kmeans(nclusters, nrows, ncolumns, data, mask, weight,
                     transpose, npass, dist, cdata, cmask, clusterid, error,
                     tclusterid, counts, mapping);

  if (npass > 1)
  { free(mapping);
    free(tclusterid);
  }

  if (transpose==0) freedatamask(nclusters, cdata, cmask);
  else freedatamask(ndata, cdata, cmask);

  free(counts);
}



void kmedoids (int nclusters, int nelements, double** distmatrix,
  int npass, int clusterid[], double* error, int* ifound)

{ int i, j, icluster;
  int* tclusterid;
  int* saved;
  int* centroids;
  double* errors;
  int ipass = 0;

  if (nelements < nclusters)
  { *ifound = 0;
    return;
  } 

  *ifound = -1;


  saved = (int*)malloc(nelements*sizeof(int));
  if (saved==NULL) return;

  centroids = (int*)malloc(nclusters*sizeof(int));
  if(!centroids)
  { free(saved);
    return;
  }

  errors = (double*)malloc(nclusters*sizeof(double));
  if(!errors)
  { free(saved);
    free(centroids);
    return;
  }

  if (npass<=1) tclusterid = clusterid;
  else
  { tclusterid = (int*)malloc(nelements*sizeof(int));
    if(!tclusterid)
    { free(saved);
      free(centroids);
      free(errors);
      return;
    }
  }

  *error = DBL_MAX;
  do 
  { double total = DBL_MAX;
    int counter = 0;
    int period = 10;

    if (npass!=0) randomassign (nclusters, nelements, tclusterid);
    while(1)
    { double previous = total;
      total = 0.0;

      if (counter % period == 0) 
      { for (i = 0; i < nelements; i++) saved[i] = tclusterid[i];
        if (period < INT_MAX / 2) period *= 2;
      }
      counter++;

   
      getclustermedoids(nclusters, nelements, distmatrix, tclusterid,
                        centroids, errors);

      for (i = 0; i < nelements; i++)
      
      { double distance = DBL_MAX;
        for (icluster = 0; icluster < nclusters; icluster++)
        { double tdistance;
          j = centroids[icluster];
          if (i==j)
          { distance = 0.0;
            tclusterid[i] = icluster;
            break;
          }
          tdistance = (i > j) ? distmatrix[i][j] : distmatrix[j][i];
          if (tdistance < distance)
          { distance = tdistance;
            tclusterid[i] = icluster;
          }
        }
        total += distance;
      }
      if (total>=previous) break;
     
      for (i = 0; i < nelements; i++)
        if (saved[i]!=tclusterid[i]) break;
      if (i==nelements)
        break; 
    }

    for (i = 0; i < nelements; i++)
    { if (clusterid[i]!=centroids[tclusterid[i]])
      { if (total < *error)
        { *ifound = 1;
          *error = total;
          
          for (j = 0; j < nelements; j++)
            clusterid[j] = centroids[tclusterid[j]];
        }
        break;
      }
    }
    if (i==nelements) (*ifound)++;
  } while (++ipass < npass);

  if (npass > 1) free(tclusterid);

  free(saved);
  free(centroids);
  free(errors);

  return;
}


double** distancematrix (int nrows, int ncolumns, double** data,
  int** mask, double weights[], char dist, int transpose)

{ 
  const int n = (transpose==0) ? nrows : ncolumns;
  const int ndata = (transpose==0) ? ncolumns : nrows;
  int i,j;
  double** matrix;

  double (*metric)
    (int, double**, double**, int**, int**, const double[], int, int, int) =
       setmetric(dist);

  if (n < 2) return NULL;

  matrix = (double**)malloc(n*sizeof(double*));
  if(matrix==NULL) return NULL; 
  matrix[0] = NULL;

  for (i = 1; i < n; i++)
  { matrix[i] = (double*)malloc(i*sizeof(double));
    if (matrix[i]==NULL) break; 
  }
  if (i < n) 
  { j = i;
    for (i = 1; i < j; i++) free(matrix[i]);
    return NULL;
  }

  for (i = 1; i < n; i++)
    for (j = 0; j < i; j++)
      matrix[i][j]=metric(ndata,data,data,mask,mask,weights,i,j,transpose);

  return matrix;
}


double* calculate_weights(int nrows, int ncolumns, double** data, int** mask,
  double weights[], int transpose, char dist, double cutoff, double exponent)


{ int i,j;
  const int ndata = (transpose==0) ? ncolumns : nrows;
  const int nelements = (transpose==0) ? nrows : ncolumns;

  
  double (*metric)
    (int, double**, double**, int**, int**, const double[], int, int, int) =
       setmetric(dist);

  double* result = (double*)malloc(nelements*sizeof(double));
  if (!result) return NULL;
  memset(result, 0, nelements*sizeof(double));

  for (i = 0; i < nelements; i++)
  { result[i] += 1.0;
    for (j = 0; j < i; j++)
    { const double distance = metric(ndata, data, data, mask, mask, weights,
                                     i, j, transpose);
      if (distance < cutoff)
      { const double dweight = exp(exponent*log(1-distance/cutoff));
     
        result[i] += dweight;
        result[j] += dweight;
      }
    }
  }
  for (i = 0; i < nelements; i++) result[i] = 1.0/result[i];
  return result;
}

void cuttree (int nelements, Node2* tree, int nclusters, int clusterid[])


{ int i, j, k;
  int icluster = 0;
  const int n = nelements-nclusters; /* number of nodes to join */
  int* nodeid;
  for (i = nelements-2; i >= n; i--)
  { k = tree[i].left;
    if (k>=0)
    { clusterid[k] = icluster;
      icluster++;
    }
    k = tree[i].right;
    if (k>=0)
    { clusterid[k] = icluster;
      icluster++;
    }
  }
  nodeid = (int*)malloc(n*sizeof(int));
  if(!nodeid)
  { for (i = 0; i < nelements; i++) clusterid[i] = -1;
    return;
  }
  for (i = 0; i < n; i++) nodeid[i] = -1;
  for (i = n-1; i >= 0; i--)
  { if(nodeid[i]<0) 
    { j = icluster;
      nodeid[i] = j;
      icluster++;
    }
    else j = nodeid[i];
    k = tree[i].left;
    if (k<0) nodeid[-k-1] = j; else clusterid[k] = j;
    k = tree[i].right;
    if (k<0) nodeid[-k-1] = j; else clusterid[k] = j;
  }
  free(nodeid);
  return;
}


static
Node2* pclcluster (int nrows, int ncolumns, double** data, int** mask,
  double weight[], double** distmatrix, char dist, int transpose)


{ int i, j;
  const int nelements = (transpose==0) ? nrows : ncolumns;
  int inode;
  const int ndata = transpose ? nrows : ncolumns;
  const int nnodes = nelements - 1;

  double (*metric)
    (int, double**, double**, int**, int**, const double[], int, int, int) =
       setmetric(dist);

  Node2* result;
  double** newdata;
  int** newmask;
  int* distid = (int*)malloc(nelements*sizeof(int));
  if(!distid) return NULL;
  result = (Node2*)malloc(nnodes*sizeof(Node2));
  if(!result)
  { free(distid);
    return NULL;
  }
  if(!makedatamask(nelements, ndata, &newdata, &newmask))
  { free(result);
    free(distid);
    return NULL; 
  }

  for (i = 0; i < nelements; i++) distid[i] = i;

  if (transpose)
  { for (i = 0; i < nelements; i++)
    { for (j = 0; j < ndata; j++)
      { newdata[i][j] = data[j][i];
        newmask[i][j] = mask[j][i];
      }
    }
    data = newdata;
    mask = newmask;
  }
  else
  { for (i = 0; i < nelements; i++)
    { memcpy(newdata[i], data[i], ndata*sizeof(double));
      memcpy(newmask[i], mask[i], ndata*sizeof(int));
    }
    data = newdata;
    mask = newmask;
  }

  for (inode = 0; inode < nnodes; inode++)
  { 
    int is = 1;
    int js = 0;
    result[inode].distance = find_closest_pair(nelements-inode, distmatrix, &is, &js);
    result[inode].left = distid[js];
    result[inode].right = distid[is];

    for (i = 0; i < ndata; i++)
    { data[js][i] = data[js][i]*mask[js][i] + data[is][i]*mask[is][i];
      mask[js][i] += mask[is][i];
      if (mask[js][i]) data[js][i] /= mask[js][i];
    }
    free(data[is]);
    free(mask[is]);
    data[is] = data[nnodes-inode];
    mask[is] = mask[nnodes-inode];
  
    distid[is] = distid[nnodes-inode];
    for (i = 0; i < is; i++)
      distmatrix[is][i] = distmatrix[nnodes-inode][i];
    for (i = is + 1; i < nnodes-inode; i++)
      distmatrix[i][is] = distmatrix[nnodes-inode][i];

    distid[js] = -inode-1;
    for (i = 0; i < js; i++)
      distmatrix[js][i] = metric(ndata,data,data,mask,mask,weight,js,i,0);
    for (i = js + 1; i < nnodes-inode; i++)
      distmatrix[i][js] = metric(ndata,data,data,mask,mask,weight,js,i,0);
  }

  free(data[0]);
  free(mask[0]);
  free(data);
  free(mask);
  free(distid);
 
  return result;
}


static
int nodecompare(const void* a, const void* b)

{ const Node2* node1 = (const Node2*)a;
  const Node2* node2 = (const Node2*)b;
  const double term1 = node1->distance;
  const double term2 = node2->distance;
  if (term1 < term2) return -1;
  if (term1 > term2) return +1;
  return 0;
}


static
Node2* pslcluster (int nrows, int ncolumns, double** data, int** mask,
  double weight[], double** distmatrix, char dist, int transpose)


{ int i, j, k;
  const int nelements = transpose ? ncolumns : nrows;
  const int nnodes = nelements - 1;
  int* vector;
  double* temp;
  int* index;
  Node2* result;
  temp = (double*)malloc(nnodes*sizeof(double));
  if(!temp) return NULL;
  index = (int*)malloc(nelements*sizeof(int));
  if(!index)
  { free(temp);
    return NULL;
  }
  vector = (int*)malloc(nnodes*sizeof(int));
  if(!vector)
  { free(index);
    free(temp);
    return NULL;
  }
  result = (Node2*)malloc(nelements*sizeof(Node2));
  if(!result)
  { free(vector);
    free(index);
    free(temp);
    return NULL;
  }

  for (i = 0; i < nnodes; i++) vector[i] = i;

  if(distmatrix)
  { for (i = 0; i < nrows; i++)
    { result[i].distance = DBL_MAX;
      for (j = 0; j < i; j++) temp[j] = distmatrix[i][j];
      for (j = 0; j < i; j++)
      { k = vector[j];
        if (result[j].distance >= temp[j])
        { if (result[j].distance < temp[k]) temp[k] = result[j].distance;
          result[j].distance = temp[j];
          vector[j] = i;
        }
        else if (temp[j] < temp[k]) temp[k] = temp[j];
      }
      for (j = 0; j < i; j++)
      {
        if (result[j].distance >= result[vector[j]].distance) vector[j] = i;
      }
    }
  }
  else
  { const int ndata = transpose ? nrows : ncolumns;
    double (*metric)
      (int, double**, double**, int**, int**, const double[], int, int, int) =
         setmetric(dist);

    for (i = 0; i < nelements; i++)
    { result[i].distance = DBL_MAX;
      for (j = 0; j < i; j++) temp[j] =
        metric(ndata, data, data, mask, mask, weight, i, j, transpose);
      for (j = 0; j < i; j++)
      { k = vector[j];
        if (result[j].distance >= temp[j])
        { if (result[j].distance < temp[k]) temp[k] = result[j].distance;
          result[j].distance = temp[j];
          vector[j] = i;
        }
        else if (temp[j] < temp[k]) temp[k] = temp[j];
      }
      for (j = 0; j < i; j++)
        if (result[j].distance >= result[vector[j]].distance) vector[j] = i;
    }
  }
  free(temp);

  for (i = 0; i < nnodes; i++) result[i].left = i;
  qsort(result, nnodes, sizeof(Node2), nodecompare);

  for (i = 0; i < nelements; i++) index[i] = i;
  for (i = 0; i < nnodes; i++)
  { j = result[i].left;
    k = vector[j];
    result[i].left = index[j];
    result[i].right = index[k];
    index[k] = -i-1;
  }
  free(vector);
  free(index);

  result = (Node2*)realloc(result, nnodes*sizeof(Node2));

  return result;
}

static Node2* pmlcluster (int nelements, double** distmatrix)

{ int j;
  int n;
  int* clusterid;
  Node2* result;

  clusterid = (int*)malloc(nelements*sizeof(int));
  if(!clusterid) return NULL;
  result = (Node2*)malloc((nelements-1)*sizeof(Node2));
  if (!result)
  { free(clusterid);
    return NULL;
  }

  for (j = 0; j < nelements; j++) clusterid[j] = j;

  for (n = nelements; n > 1; n--)
  { int is = 1;
    int js = 0;
    result[nelements-n].distance = find_closest_pair(n, distmatrix, &is, &js);

    for (j = 0; j < js; j++)
      distmatrix[js][j] = max(distmatrix[is][j],distmatrix[js][j]);
    for (j = js+1; j < is; j++)
      distmatrix[j][js] = max(distmatrix[is][j],distmatrix[j][js]);
    for (j = is+1; j < n; j++)
      distmatrix[j][js] = max(distmatrix[j][is],distmatrix[j][js]);

    for (j = 0; j < is; j++) distmatrix[is][j] = distmatrix[n-1][j];
    for (j = is+1; j < n-1; j++) distmatrix[j][is] = distmatrix[n-1][j];

    result[nelements-n].left = clusterid[is];
    result[nelements-n].right = clusterid[js];
    clusterid[js] = n-nelements-1;
    clusterid[is] = clusterid[n-1];
  }
  free(clusterid);

  return result;
}


static Node2* palcluster (int nelements, double** distmatrix)

{ int j;
  int n;
  int* clusterid;
  int* number;
  Node2* result;

  clusterid = (int*)malloc(nelements*sizeof(int));
  if(!clusterid) return NULL;
  number = (int*)malloc(nelements*sizeof(int));
  if(!number)
  { free(clusterid);
    return NULL;
  }
  result = (Node2*)malloc((nelements-1)*sizeof(Node2));
  if (!result)
  { free(clusterid);
    free(number);
    return NULL;
  }

  for (j = 0; j < nelements; j++)
  { number[j] = 1;
    clusterid[j] = j;
  }

  for (n = nelements; n > 1; n--)
  { int sum;
    int is = 1;
    int js = 0;
    result[nelements-n].distance = find_closest_pair(n, distmatrix, &is, &js);

    result[nelements-n].left = clusterid[is];
    result[nelements-n].right = clusterid[js];

    sum = number[is] + number[js];
    for (j = 0; j < js; j++)
    { distmatrix[js][j] = distmatrix[is][j]*number[is]
                        + distmatrix[js][j]*number[js];
      distmatrix[js][j] /= sum;
    }
    for (j = js+1; j < is; j++)
    { distmatrix[j][js] = distmatrix[is][j]*number[is]
                        + distmatrix[j][js]*number[js];
      distmatrix[j][js] /= sum;
    }
    for (j = is+1; j < n; j++)
    { distmatrix[j][js] = distmatrix[j][is]*number[is]
                        + distmatrix[j][js]*number[js];
      distmatrix[j][js] /= sum;
    }

    for (j = 0; j < is; j++) distmatrix[is][j] = distmatrix[n-1][j];
    for (j = is+1; j < n-1; j++) distmatrix[j][is] = distmatrix[n-1][j];

    number[js] = sum;
    number[is] = number[n-1];

    clusterid[js] = n-nelements-1;
    clusterid[is] = clusterid[n-1];
  }
  free(clusterid);
  free(number);

  return result;
}

Node2* treecluster (int nrows, int ncolumns, double** data, int** mask,
  double weight[], int transpose, char dist, char method, double** distmatrix)

{ Node2* result = NULL;
  const int nelements = (transpose==0) ? nrows : ncolumns;
  const int ldistmatrix = (distmatrix==NULL && method!='s') ? 1 : 0;

  if (nelements < 2) return NULL;

  if(ldistmatrix)
  { distmatrix =
      distancematrix(nrows, ncolumns, data, mask, weight, dist, transpose);
    if (!distmatrix) return NULL;
  }

  switch(method)
  { case 's':
      result = pslcluster(nrows, ncolumns, data, mask, weight, distmatrix,
                          dist, transpose);
      break;
    case 'm':
      result = pmlcluster(nelements, distmatrix);
      break;
    case 'a':
      result = palcluster(nelements, distmatrix);
      break;
    case 'c':
      result = pclcluster(nrows, ncolumns, data, mask, weight, distmatrix,
                          dist, transpose);
      break;
  }

  if(ldistmatrix)
  { int i;
    for (i = 1; i < nelements; i++) free(distmatrix[i]);
    free (distmatrix);
  }
 
  return result;
}


static
void somworker (int nrows, int ncolumns, double** data, int** mask,
  const double weights[], int transpose, int nxgrid, int nygrid,
  double inittau, double*** celldata, int niter, char dist)

{ const int nelements = (transpose==0) ? nrows : ncolumns;
  const int ndata = (transpose==0) ? ncolumns : nrows;
  int i, j;
  double* stddata = (double*)calloc(nelements,sizeof(double));
  int** dummymask;
  int ix, iy;
  int* index;
  int iter;
  double maxradius = sqrt((double)nxgrid*nxgrid+nygrid*nygrid);

  double (*metric)
    (int, double**, double**, int**, int**, const double[], int, int, int) =
       setmetric(dist);

  if (transpose==0)
  { for (i = 0; i < nelements; i++)
    { int n = 0;
      for (j = 0; j < ndata; j++)
      { if (mask[i][j])
        { double term = data[i][j];
          term = term * term;
          stddata[i] += term;
          n++;
        }
      }
      if (stddata[i] > 0) stddata[i] = sqrt(stddata[i]/n);
      else stddata[i] = 1;
    }
  }
  else
  { for (i = 0; i < nelements; i++)
    { int n = 0;
      for (j = 0; j < ndata; j++)
      { if (mask[j][i])
        { double term = data[j][i];
          term = term * term;
          stddata[i] += term;
          n++;
        }
      }
      if (stddata[i] > 0) stddata[i] = sqrt(stddata[i]/n);
      else stddata[i] = 1;
    }
  }

  if (transpose==0)
  { dummymask = (int**)malloc(nygrid*sizeof(int*));
    for (i = 0; i < nygrid; i++)
    { dummymask[i] = (int*)malloc(ndata*sizeof(int));
      for (j = 0; j < ndata; j++) dummymask[i][j] = 1;
    }
  }
  else
  { dummymask = (int**)malloc(ndata*sizeof(int*));
    for (i = 0; i < ndata; i++)
    { dummymask[i] = (int*)malloc(sizeof(int));
      dummymask[i][0] = 1;
    }
  }

  for (ix = 0; ix < nxgrid; ix++)
  { for (iy = 0; iy < nygrid; iy++)
    { double sum = 0.;
      for (i = 0; i < ndata; i++)
      { double term = -1.0 + 2.0*uniform();
        celldata[ix][iy][i] = term;
        sum += term * term;
      }
      sum = sqrt(sum/ndata);
      for (i = 0; i < ndata; i++) celldata[ix][iy][i] /= sum;
    }
  }

  index = (int*)malloc(nelements*sizeof(int));
  for (i = 0; i < nelements; i++) index[i] = i;
  for (i = 0; i < nelements; i++)
  { j = (int) (i + (nelements-i)*uniform());
    ix = index[j];
    index[j] = index[i];
    index[i] = ix;
  }

  for (iter = 0; iter < niter; iter++)
  { int ixbest = 0;
    int iybest = 0;
    int iobject = iter % nelements;
    iobject = index[iobject];
    if (transpose==0)
    { double closest = metric(ndata,data,celldata[ixbest],
        mask,dummymask,weights,iobject,iybest,transpose);
      double radius = maxradius * (1. - ((double)iter)/((double)niter));
      double tau = inittau * (1. - ((double)iter)/((double)niter));

      for (ix = 0; ix < nxgrid; ix++)
      { for (iy = 0; iy < nygrid; iy++)
        { double distance =
            metric (ndata,data,celldata[ix],
              mask,dummymask,weights,iobject,iy,transpose);
          if (distance < closest)
          { ixbest = ix;
            iybest = iy;
            closest = distance;
          }
        }
      }
      for (ix = 0; ix < nxgrid; ix++)
      { for (iy = 0; iy < nygrid; iy++)
        { if (sqrt((double)(ix-ixbest)*(ix-ixbest)+(iy-iybest)*(iy-iybest))<radius)
          { double sum = 0.;
            for (i = 0; i < ndata; i++)
            { if (mask[iobject][i]==0) continue;
              celldata[ix][iy][i] +=
                tau * (data[iobject][i]/stddata[iobject]-celldata[ix][iy][i]);
            }
            for (i = 0; i < ndata; i++)
            { double term = celldata[ix][iy][i];
              term = term * term;
              sum += term;
            }
            if (sum>0)
            { sum = sqrt(sum/ndata);
              for (i = 0; i < ndata; i++) celldata[ix][iy][i] /= sum;
            }
          }
        }
      }
    }
    else
    { double closest;
      double** celldatavector = (double**)malloc(ndata*sizeof(double*));
      double radius = maxradius * (1. - ((double)iter)/((double)niter));
      double tau = inittau * (1. - ((double)iter)/((double)niter));

      for (i = 0; i < ndata; i++)
        celldatavector[i] = &(celldata[ixbest][iybest][i]);
      closest = metric(ndata,data,celldatavector,
        mask,dummymask,weights,iobject,0,transpose);
      for (ix = 0; ix < nxgrid; ix++)
      { for (iy = 0; iy < nygrid; iy++)
        { double distance;
          for (i = 0; i < ndata; i++)
            celldatavector[i] = &(celldata[ixbest][iybest][i]);
          distance =
            metric (ndata,data,celldatavector,
              mask,dummymask,weights,iobject,0,transpose);
          if (distance < closest)
          { ixbest = ix;
            iybest = iy;
            closest = distance;
          }
        }
      }
      free(celldatavector);
      for (ix = 0; ix < nxgrid; ix++)
      { for (iy = 0; iy < nygrid; iy++)
        { if (sqrt((double)(ix-ixbest)*(ix-ixbest)+(iy-iybest)*(iy-iybest))<radius)
          { double sum = 0.;
            for (i = 0; i < ndata; i++)
            { if (mask[i][iobject]==0) continue;
              celldata[ix][iy][i] +=
                tau * (data[i][iobject]/stddata[iobject]-celldata[ix][iy][i]);
            }
            for (i = 0; i < ndata; i++)
            { double term = celldata[ix][iy][i];
              term = term * term;
              sum += term;
            }
            if (sum>0)
            { sum = sqrt(sum/ndata);
              for (i = 0; i < ndata; i++) celldata[ix][iy][i] /= sum;
            }
          }
        }
      }
    }
  }
  if (transpose==0)
    for (i = 0; i < nygrid; i++) free(dummymask[i]);
  else
    for (i = 0; i < ndata; i++) free(dummymask[i]);
  free(dummymask);
  free(stddata);
  free(index);
  return;
}

static
void somassign (int nrows, int ncolumns, double** data, int** mask,
  const double weights[], int transpose, int nxgrid, int nygrid,
  double*** celldata, char dist, int clusterid[][2])

{ const int ndata = (transpose==0) ? ncolumns : nrows;
  int i,j;

  double (*metric)
    (int, double**, double**, int**, int**, const double[], int, int, int) =
       setmetric(dist);

  if (transpose==0)
  { int** dummymask = (int**)malloc(nygrid*sizeof(int*));
    for (i = 0; i < nygrid; i++)
    { dummymask[i] = (int*)malloc(ncolumns*sizeof(int));
      for (j = 0; j < ncolumns; j++) dummymask[i][j] = 1;
    }
    for (i = 0; i < nrows; i++)
    { int ixbest = 0;
      int iybest = 0;
      double closest = metric(ndata,data,celldata[ixbest],
        mask,dummymask,weights,i,iybest,transpose);
      int ix, iy;
      for (ix = 0; ix < nxgrid; ix++)
      { for (iy = 0; iy < nygrid; iy++)
        { double distance =
            metric (ndata,data,celldata[ix],
              mask,dummymask,weights,i,iy,transpose);
          if (distance < closest)
          { ixbest = ix;
            iybest = iy;
            closest = distance;
          }
        }
      }
      clusterid[i][0] = ixbest;
      clusterid[i][1] = iybest;
    }
    for (i = 0; i < nygrid; i++) free(dummymask[i]);
    free(dummymask);
  }
  else
  { double** celldatavector = (double**)malloc(ndata*sizeof(double*));
    int** dummymask = (int**)malloc(nrows*sizeof(int*));
    int ixbest = 0;
    int iybest = 0;
    for (i = 0; i < nrows; i++)
    { dummymask[i] = (int*)malloc(sizeof(int));
      dummymask[i][0] = 1;
    }
    for (i = 0; i < ncolumns; i++)
    { double closest;
      int ix, iy;
      for (j = 0; j < ndata; j++)
        celldatavector[j] = &(celldata[ixbest][iybest][j]);
      closest = metric(ndata,data,celldatavector,
        mask,dummymask,weights,i,0,transpose);
      for (ix = 0; ix < nxgrid; ix++)
      { for (iy = 0; iy < nygrid; iy++)
        { double distance;
          for(j = 0; j < ndata; j++)
            celldatavector[j] = &(celldata[ix][iy][j]);
          distance = metric(ndata,data,celldatavector,
            mask,dummymask,weights,i,0,transpose);
          if (distance < closest)
          { ixbest = ix;
            iybest = iy;
            closest = distance;
          }
        }
      }
      clusterid[i][0] = ixbest;
      clusterid[i][1] = iybest;
    }
    free(celldatavector);
    for (i = 0; i < nrows; i++) free(dummymask[i]);
    free(dummymask);
  }
  return;
}



void somcluster (int nrows, int ncolumns, double** data, int** mask,
  const double weight[], int transpose, int nxgrid, int nygrid,
  double inittau, int niter, char dist, double*** celldata, int clusterid[][2])

{ const int nobjects = (transpose==0) ? nrows : ncolumns;
  const int ndata = (transpose==0) ? ncolumns : nrows;
  int i,j;
  const int lcelldata = (celldata==NULL) ? 0 : 1;

  if (nobjects < 2) return;

  if (lcelldata==0)
  { celldata = (double***)malloc(nxgrid*nygrid*ndata*sizeof(double**));
    for (i = 0; i < nxgrid; i++)
    { celldata[i] = (double**)malloc(nygrid*ndata*sizeof(double*));
      for (j = 0; j < nygrid; j++)
        celldata[i][j] = (double*)malloc(ndata*sizeof(double));
    }
  }

  somworker (nrows, ncolumns, data, mask, weight, transpose, nxgrid, nygrid,
    inittau, celldata, niter, dist);
  if (clusterid)
    somassign (nrows, ncolumns, data, mask, weight, transpose,
      nxgrid, nygrid, celldata, dist, clusterid);
  if(lcelldata==0)
  { for (i = 0; i < nxgrid; i++)
      for (j = 0; j < nygrid; j++)
        free(celldata[i][j]);
    for (i = 0; i < nxgrid; i++)
      free(celldata[i]);
    free(celldata);
  }
  return;
}


double clusterdistance (int nrows, int ncolumns, double** data,
  int** mask, double weight[], int n1, int n2, int index1[], int index2[],
  char dist, char method, int transpose)
              

{ 
  double (*metric)
    (int, double**, double**, int**, int**, const double[], int, int, int) =
       setmetric(dist);

  if (n1 < 1 || n2 < 1) return -1.0;

  if (transpose==0)
  { int i;
    for (i = 0; i < n1; i++)
    { int index = index1[i];
      if (index < 0 || index >= nrows) return -1.0;
    }
    for (i = 0; i < n2; i++)
    { int index = index2[i];
      if (index < 0 || index >= nrows) return -1.0;
    }
  }
  else
  { int i;
    for (i = 0; i < n1; i++)
    { int index = index1[i];
      if (index < 0 || index >= ncolumns) return -1.0;
    }
    for (i = 0; i < n2; i++)
    { int index = index2[i];
      if (index < 0 || index >= ncolumns) return -1.0;
    }
  }

  switch (method)
  { case 'a':
    { 
      int i,j,k;
      if (transpose==0)
      { double distance;
        double* cdata[2];
        int* cmask[2];
        int* count[2];
        count[0] = (int*)calloc(ncolumns,sizeof(int));
        count[1] = (int*)calloc(ncolumns,sizeof(int));
        cdata[0] = (double*)calloc(ncolumns,sizeof(double));
        cdata[1] = (double*)calloc(ncolumns,sizeof(double));
        cmask[0] = (int*)malloc(ncolumns*sizeof(int));
        cmask[1] = (int*)malloc(ncolumns*sizeof(int));
        for (i = 0; i < n1; i++)
        { k = index1[i];
          for (j = 0; j < ncolumns; j++)
            if (mask[k][j] != 0)
            { cdata[0][j] = cdata[0][j] + data[k][j];
              count[0][j] = count[0][j] + 1;
            }
        }
        for (i = 0; i < n2; i++)
        { k = index2[i];
          for (j = 0; j < ncolumns; j++)
            if (mask[k][j] != 0)
            { cdata[1][j] = cdata[1][j] + data[k][j];
              count[1][j] = count[1][j] + 1;
            }
        }
        for (i = 0; i < 2; i++)
          for (j = 0; j < ncolumns; j++)
          { if (count[i][j]>0)
            { cdata[i][j] = cdata[i][j] / count[i][j];
              cmask[i][j] = 1;
            }
            else
              cmask[i][j] = 0;
          }
        distance =
          metric (ncolumns,cdata,cdata,cmask,cmask,weight,0,1,0);
        for (i = 0; i < 2; i++)
        { free (cdata[i]);
          free (cmask[i]);
          free (count[i]);
        }
        return distance;
      }
      else
      { double distance;
        int** count = (int**)malloc(nrows*sizeof(int*));
        double** cdata = (double**)malloc(nrows*sizeof(double*));
        int** cmask = (int**)malloc(nrows*sizeof(int*));
        for (i = 0; i < nrows; i++)
        { count[i] = (int*)calloc(2,sizeof(int));
          cdata[i] = (double*)calloc(2,sizeof(double));
          cmask[i] = (int*)malloc(2*sizeof(int));
        }
        for (i = 0; i < n1; i++)
        { k = index1[i];
          for (j = 0; j < nrows; j++)
          { if (mask[j][k] != 0)
            { cdata[j][0] = cdata[j][0] + data[j][k];
              count[j][0] = count[j][0] + 1;
            }
          }
        }
        for (i = 0; i < n2; i++)
        { k = index2[i];
          for (j = 0; j < nrows; j++)
          { if (mask[j][k] != 0)
            { cdata[j][1] = cdata[j][1] + data[j][k];
              count[j][1] = count[j][1] + 1;
            }
          }
        }
        for (i = 0; i < nrows; i++)
          for (j = 0; j < 2; j++)
            if (count[i][j]>0)
            { cdata[i][j] = cdata[i][j] / count[i][j];
              cmask[i][j] = 1;
            }
            else
              cmask[i][j] = 0;
        distance = metric (nrows,cdata,cdata,cmask,cmask,weight,0,1,1);
        for (i = 0; i < nrows; i++)
        { free (count[i]);
          free (cdata[i]);
          free (cmask[i]);
        }
        free (count);
        free (cdata);
        free (cmask);
        return distance;
      }
    }
    case 'm':
    { int i, j, k;
      if (transpose==0)
      { double distance;
        double* temp = (double*)malloc(nrows*sizeof(double));
        double* cdata[2];
        int* cmask[2];
        for (i = 0; i < 2; i++)
        { cdata[i] = (double*)malloc(ncolumns*sizeof(double));
          cmask[i] = (int*)malloc(ncolumns*sizeof(int));
        }
        for (j = 0; j < ncolumns; j++)
        { int count = 0;
          for (k = 0; k < n1; k++)
          { i = index1[k];
            if (mask[i][j])
            { temp[count] = data[i][j];
              count++;
            }
          }
          if (count>0)
          { cdata[0][j] = median (count,temp);
            cmask[0][j] = 1;
          }
          else
          { cdata[0][j] = 0.;
            cmask[0][j] = 0;
          }
        }
        for (j = 0; j < ncolumns; j++)
        { int count = 0;
          for (k = 0; k < n2; k++)
          { i = index2[k];
            if (mask[i][j])
            { temp[count] = data[i][j];
              count++;
            }
          }
          if (count>0)
          { cdata[1][j] = median (count,temp);
            cmask[1][j] = 1;
          }
          else
          { cdata[1][j] = 0.;
            cmask[1][j] = 0;
          }
        }
        distance = metric (ncolumns,cdata,cdata,cmask,cmask,weight,0,1,0);
        for (i = 0; i < 2; i++)
        { free (cdata[i]);
          free (cmask[i]);
        }
        free(temp);
        return distance;
      }
      else
      { double distance;
        double* temp = (double*)malloc(ncolumns*sizeof(double));
        double** cdata = (double**)malloc(nrows*sizeof(double*));
        int** cmask = (int**)malloc(nrows*sizeof(int*));
        for (i = 0; i < nrows; i++)
        { cdata[i] = (double*)malloc(2*sizeof(double));
          cmask[i] = (int*)malloc(2*sizeof(int));
        }
        for (j = 0; j < nrows; j++)
        { int count = 0;
          for (k = 0; k < n1; k++)
          { i = index1[k];
            if (mask[j][i])
            { temp[count] = data[j][i];
              count++;
            }
          }
          if (count>0)
          { cdata[j][0] = median (count,temp);
            cmask[j][0] = 1;
          }
          else
          { cdata[j][0] = 0.;
            cmask[j][0] = 0;
          }
        }
        for (j = 0; j < nrows; j++)
        { int count = 0;
          for (k = 0; k < n2; k++)
          { i = index2[k];
            if (mask[j][i])
            { temp[count] = data[j][i];
              count++;
            }
          }
          if (count>0)
          { cdata[j][1] = median (count,temp);
            cmask[j][1] = 1;
          }
          else
          { cdata[j][1] = 0.;
            cmask[j][1] = 0;
          }
        }
        distance = metric (nrows,cdata,cdata,cmask,cmask,weight,0,1,1);
        for (i = 0; i < nrows; i++)
        { free (cdata[i]);
          free (cmask[i]);
        }
        free(cdata);
        free(cmask);
        free(temp);
        return distance;
      }
    }
    case 's':
    { int i1, i2, j1, j2;
      const int n = (transpose==0) ? ncolumns : nrows;
      double mindistance = DBL_MAX;
      for (i1 = 0; i1 < n1; i1++)
        for (i2 = 0; i2 < n2; i2++)
        { double distance;
          j1 = index1[i1];
          j2 = index2[i2];
          distance = metric (n,data,data,mask,mask,weight,j1,j2,transpose);
          if (distance < mindistance) mindistance = distance;
        }
      return mindistance;
    }
    case 'x':
    { int i1, i2, j1, j2;
      const int n = (transpose==0) ? ncolumns : nrows;
      double maxdistance = 0;
      for (i1 = 0; i1 < n1; i1++)
        for (i2 = 0; i2 < n2; i2++)
        { double distance;
          j1 = index1[i1];
          j2 = index2[i2];
          distance = metric (n,data,data,mask,mask,weight,j1,j2,transpose);
          if (distance > maxdistance) maxdistance = distance;
        }
      return maxdistance;
    }
    case 'v':
    { int i1, i2, j1, j2;
      const int n = (transpose==0) ? ncolumns : nrows;
      double distance = 0;
      for (i1 = 0; i1 < n1; i1++)
        for (i2 = 0; i2 < n2; i2++)
        { j1 = index1[i1];
          j2 = index2[i2];
          distance += metric (n,data,data,mask,mask,weight,j1,j2,transpose);
        }
      distance /= (n1*n2);
      return distance;
    }
  }

  return -2.0;
}
