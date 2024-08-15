# include <cmath>
# include <cstdlib>
# include <iomanip>
# include <iostream>
# include <ctime>

using namespace std;

# include "halton.hpp"
# include "faure.hpp"

//****************************************************************************80

double *halton ( int i, int m )

//****************************************************************************80
//
//  Purpose:
//
//    HALTON computes an element of a Halton sequence.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    12 August 2016
//
//  Author:
//
//    John Burkardt
//
//  Reference:
//
//    John Halton,
//    On the efficiency of certain quasi-random sequences of points
//    in evaluating multi-dimensional integrals,
//    Numerische Mathematik,
//    Volume 2, pages 84-90, 1960.
//
//  Parameters:
//
//    Input, int I, the index of the element of the sequence.
//    0 <= I.
//
//    Input, int M, the spatial dimension.
//
//    Output, double HALTON[M], the element of the sequence with index I.
//
{
  int d;
  int j;
  double *prime_inv;
  double *r;
  int *t;

  prime_inv = new double[m];
  r = new double[m];
  t = new int[m];

  for ( j = 0; j < m; j++ )
  {
    t[j] = i;
  }
//
//  Carry out the computation.
//
  for ( j = 0; j < m; j++ )
  {
    prime_inv[j] = 1.0 / ( double ) ( prime ( j + 1 ) );
  }

  for ( j = 0; j < m; j++ )
  {
    r[j] = 0.0;
  }

  while ( 0 < i4vec_sum ( m, t ) )
  {
    for ( j = 0; j < m; j++ )
    {
      d = ( t[j] % prime ( j + 1 ) );
      r[j] = r[j] + ( double ) ( d ) * prime_inv[j];
      prime_inv[j] = prime_inv[j] / ( double ) ( prime ( j + 1 ) );
      t[j] = ( t[j] / prime ( j + 1 ) );
    }
  }

  delete [] prime_inv;
  delete [] t;

  return r;
}
//****************************************************************************80

double *halton_base ( int i, int m, int b[] )

//****************************************************************************80
//
//  Purpose:
//
//    HALTON_BASE computes an element of a Halton sequence with user bases.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    12 August 2016
//
//  Author:
//
//    John Burkardt
//
//  Reference:
//
//    John Halton,
//    On the efficiency of certain quasi-random sequences of points
//    in evaluating multi-dimensional integrals,
//    Numerische Mathematik,
//    Volume 2, pages 84-90, 1960.
//
//  Parameters:
//
//    Input, int I, the index of the element of the sequence.
//    0 <= I.
//
//    Input, int M, the spatial dimension.
//
//    Input, int B[M], the bases to use for each dimension.
//
//    Output, double HALTON_BASE[M], the element of the sequence with index I.
//
{
  double *b_inv;
  int d;
  int j;
  double *r;
  int *t;

  b_inv = new double[m];
  r = new double[m];
  t = new int[m];

  for ( j = 0; j < m; j++ )
  {
    t[j] = i;
  }
//
//  Carry out the computation.
//
  for ( j = 0; j < m; j++ )
  {
    b_inv[j] = 1.0 / ( double ) ( b[j] );
  }

  for ( j = 0; j < m; j++ )
  {
    r[j] = 0.0;
  }

  while ( 0 < i4vec_sum ( m, t ) )
  {
    for ( j = 0; j < m; j++ )
    {
      d = ( t[j] % b[j] );
      r[j] = r[j] + ( double ) ( d ) * b_inv[j];
      b_inv[j] = b_inv[j] / ( double ) ( b[j] );
      t[j] = ( t[j] / b[j] );
    }
  }

  delete [] b_inv;
  delete [] t;

  return r;
}
//****************************************************************************80

int halton_inverse ( double r[], int m )

//****************************************************************************80
//
//  Purpose:
//
//    HALTON_INVERSE inverts an element of the Halton sequence.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    12 August 2016
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, double R[M], the I-th element of the Halton sequence.
//    0 <= R < 1.0
//
//    Input, int M, the spatial dimension.
//
//    Output, int HALTON_INVERSE, the index of the element of the sequence.
//
{
  int d;
  int i;
  int j;
  int p;
  double t;

  for ( j = 0; j < m; j++ )
  {
    if ( r[j] < 0.0 || 1.0 <= r[j] )
    {
      cerr << "\n";
      cerr << "HALTON_INVERSE - Fatal error\n";
      cerr << "  0 <= R < 1.0 is required.\n";
      exit ( 1 );
    }
  }
//
//  Invert using the first component only, because working with base
//  2 is accurate.
//
  i = 0;
  t = r[0];
  p = 1;

  while ( t != 0.0 )
  {
    t = t * 2.0;
    d = ( int ) ( t );
    i = i + d * p;
    p = p * 2;
    t = r8_mod ( t, 1.0 );
  }
 
  return i;
}
//****************************************************************************80

double *halton_sequence ( int i1, int i2, int m )

//****************************************************************************80
//
//  Purpose:
//
//    HALTON_SEQUENCE computes elements I1 through I2 of a Halton sequence.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    19 August 2016
//
//  Author:
//
//    John Burkardt
//
//  Reference:
//
//    John Halton,
//    On the efficiency of certain quasi-random sequences of points
//    in evaluating multi-dimensional integrals,
//    Numerische Mathematik,
//    Volume 2, pages 84-90, 1960.
//
//  Parameters:
//
//    Input, int I1, I2, the indices of the first and last 
//    elements of the sequence.  0 <= I1, I2.
//
//    Input, int M, the spatial dimension.
//    1 <= M <= 100.
//
//    Output, double HALTON_SEQUENCE[M*(abs(I1-I2)+1)], the elements of the 
//    sequence with indices I1 through I2.
//
{
  int d;
  int i;
  int i3;
  int j;
  int k;
  int n;
  double *prime_inv;
  double *r;
  int *t;

  prime_inv = new double[m];
  r = new double[m*(abs(i1-i2)+1)];
  t = new int[m];

  if ( i1 <= i2 )
  {
    i3 = +1;
  }
  else
  {
    i3 = -1;
  }

  n = abs ( i2 - i1 ) + 1;

  for ( j = 0; j < n; j++ )
  {
    for ( i = 0; i < m; i++ )
    {
      r[i+j*m] = 0.0;
    }
  }

  i = i1;

  for ( k = 0; k < n; k++ )
  {
    for ( j = 0; j < m; j++ )
    {
      t[j] = i;
    }
//
//  Carry out the computation.
//
    for ( j = 0; j < m; j++ )
    {
      prime_inv[j] = 1.0 / ( double ) ( prime ( j + 1 ) );
    }

    while ( 0 < i4vec_sum ( m, t ) )
    {
      for ( j = 0; j < m; j++ )
      {
        d = ( t[j] % prime ( j + 1 ) );
        r[j+k*m] = r[j+k*m] + ( double ) ( d ) * prime_inv[j];
        prime_inv[j] = prime_inv[j] / ( double ) ( prime ( j + 1 ) );
        t[j] = ( t[j] / prime ( j + 1 ) );
      }
    }
    i = i + i3;
  }

  delete [] prime_inv;
  delete [] t;

  return r;
}
//****************************************************************************80

int i4vec_sum ( int n, int a[] )

//****************************************************************************80
//
//  Purpose:
//
//    I4VEC_SUM sums the entries of an I4VEC.
//
//  Discussion:
//
//    An I4VEC is a vector of I4's.
//
//  Example:
//
//    Input:
//
//      A = ( 1, 2, 3, 4 )
//
//    Output:
//
//      I4VEC_SUM = 10
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    26 May 1999
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, int N, the number of entries in the vector.
//
//    Input, int A[N], the vector to be summed.
//
//    Output, int I4VEC_SUM, the sum of the entries of A.
//
{
  int i;
  int sum;

  sum = 0;
  for ( i = 0; i < n; i++ )
  {
    sum = sum + a[i];
  }

  return sum;
}
//****************************************************************************80

double r8_mod ( double x, double y )

//****************************************************************************80
//
//  Purpose:
//
//    R8_MOD returns the remainder of R8 division.
//
//  Discussion:
//
//    If
//      REM = R8_MOD ( X, Y )
//      RMULT = ( X - REM ) / Y
//    then
//      X = Y * RMULT + REM
//    where REM has the same sign as X, and abs ( REM ) < Y.
//
//  Example:
//
//        X         Y     R8_MOD   R8_MOD  Factorization
//
//      107        50       7     107 =  2 *  50 + 7
//      107       -50       7     107 = -2 * -50 + 7
//     -107        50      -7    -107 = -2 *  50 - 7
//     -107       -50      -7    -107 =  2 * -50 - 7
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    14 June 2007
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, double X, the number to be divided.
//
//    Input, double Y, the number that divides X.
//
//    Output, double R8_MOD, the remainder when X is divided by Y.
//
{
  double value;

  if ( y == 0.0 )
  {
    cerr << "\n";
    cerr << "R8_MOD - Fatal error!\n";
    cerr << "  R8_MOD ( X, Y ) called with Y = " << y << "\n";
    exit ( 1 );
  }

  value = x - ( ( double ) ( ( int ) ( x / y ) ) ) * y;

  if ( x < 0.0 && 0.0 < value )
  {
    value = value - fabs ( y );
  }
  else if ( 0.0 < x && value < 0.0 )
  {
    value = value + fabs ( y );
  }

  return value;
}
//****************************************************************************80

// void r8mat_print ( int m, int n, double a[], string title )

// //****************************************************************************80
// //
// //  Purpose:
// //
// //    R8MAT_PRINT prints an R8MAT.
// //
// //  Discussion:
// //
// //    An R8MAT is a doubly dimensioned array of R8 values, stored as a vector
// //    in column-major order.
// //
// //    Entry A(I,J) is stored as A[I+J*M]
// //
// //  Licensing:
// //
// //    This code is distributed under the GNU LGPL license.
// //
// //  Modified:
// //
// //    10 September 2009
// //
// //  Author:
// //
// //    John Burkardt
// //
// //  Parameters:
// //
// //    Input, int M, the number of rows in A.
// //
// //    Input, int N, the number of columns in A.
// //
// //    Input, double A[M*N], the M by N matrix.
// //
// //    Input, string TITLE, a title.
// //
// {
//   r8mat_print_some ( m, n, a, 1, 1, m, n, title );

//   return;
// }
// //****************************************************************************80

// void r8mat_print_some ( int m, int n, double a[], int ilo, int jlo, int ihi,
//   int jhi, string title )

// //****************************************************************************80
// //
// //  Purpose:
// //
// //    R8MAT_PRINT_SOME prints some of an R8MAT.
// //
// //  Discussion:
// //
// //    An R8MAT is a doubly dimensioned array of R8 values, stored as a vector
// //    in column-major order.
// //
// //  Licensing:
// //
// //    This code is distributed under the GNU LGPL license.
// //
// //  Modified:
// //
// //    26 June 2013
// //
// //  Author:
// //
// //    John Burkardt
// //
// //  Parameters:
// //
// //    Input, int M, the number of rows of the matrix.
// //    M must be positive.
// //
// //    Input, int N, the number of columns of the matrix.
// //    N must be positive.
// //
// //    Input, double A[M*N], the matrix.
// //
// //    Input, int ILO, JLO, IHI, JHI, designate the first row and
// //    column, and the last row and column to be printed.
// //
// //    Input, string TITLE, a title.
// //
// {
// # define INCX 5

//   int i;
//   int i2hi;
//   int i2lo;
//   int j;
//   int j2hi;
//   int j2lo;

//   cout << "\n";
//   cout << title << "\n";

//   if ( m <= 0 || n <= 0 )
//   {
//     cout << "\n";
//     cout << "  (None)\n";
//     return;
//   }
// //
// //  Print the columns of the matrix, in strips of 5.
// //
//   for ( j2lo = jlo; j2lo <= jhi; j2lo = j2lo + INCX )
//   {
//     j2hi = j2lo + INCX - 1;
//     if ( n < j2hi )
//     {
//       j2hi = n;
//     }
//     if ( jhi < j2hi )
//     {
//       j2hi = jhi;
//     }
//     cout << "\n";
// //
// //  For each column J in the current range...
// //
// //  Write the header.
// //
//     cout << "  Col:    ";
//     for ( j = j2lo; j <= j2hi; j++ )
//     {
//       cout << setw(7) << j - 1 << "       ";
//     }
//     cout << "\n";
//     cout << "  Row\n";
//     cout << "\n";
// //
// //  Determine the range of the rows in this strip.
// //
//     if ( 1 < ilo )
//     {
//       i2lo = ilo;
//     }
//     else
//     {
//       i2lo = 1;
//     }
//     if ( ihi < m )
//     {
//       i2hi = ihi;
//     }
//     else
//     {
//       i2hi = m;
//     }

//     for ( i = i2lo; i <= i2hi; i++ )
//     {
// //
// //  Print out (up to) 5 entries in row I, that lie in the current strip.
// //
//       cout << setw(5) << i - 1 << ": ";
//       for ( j = j2lo; j <= j2hi; j++ )
//       {
//         cout << setw(12) << a[i-1+(j-1)*m] << "  ";
//       }
//       cout << "\n";
//     }
//   }

//   return;
// # undef INCX
// }
// //****************************************************************************80

void timestamp ( )

//****************************************************************************80
//
//  Purpose:
//
//    TIMESTAMP prints the current YMDHMS date as a time stamp.
//
//  Example:
//
//    31 May 2001 09:45:54 AM
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    08 July 2009
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    None
//
{
# define TIME_SIZE 40

  static char time_buffer[TIME_SIZE];
  const struct std::tm *tm_ptr;
  std::time_t now;

  now = std::time ( NULL );
  tm_ptr = std::localtime ( &now );

  std::strftime ( time_buffer, TIME_SIZE, "%d %B %Y %I:%M:%S %p", tm_ptr );

  std::cout << time_buffer << "\n";

  return;
# undef TIME_SIZE
}
