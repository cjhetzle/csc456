__kernel void merge(__global float* restrict data, int l, int m,
  int r, __local float* restrict tmp)
{
  int i = l;
  int j = m;
  int k = r;

  while ( i < m || j < r ) {
    if ( i < m && j < r ) { 
       if ( data[i] < data[j] )
          tmp[k++] = data[i++];
       else
          tmp[k++] = data[j++];
    }
    else if ( i == m )
       tmp[k++] = data[j++]; 
    else if ( j == r )
       tmp[k++] = data[i++]; 
  }

  for ( i = l; i < r; i++ )
    data[i] = tmp[i];
}

__kernel void mergeSort(__global float* restrict data,
   __local float* restrict tmp, const int r)
{
  int width;

  for ( width = 1; width < r; width = 2*width ) {
    //combine arrays of length 'width'
    int i;
    for ( i = 0; i < r; i = i + 2*width ) {
      int left, right, middle;

      left = i;
      middle = i + width;
      right  = i + 2*width;

      merge( data, left, middle, right, tmp );
    }
  }
}

__kernel void fpgasort(__global float* restrict x,  
                        const int size)
{
    // get index of the work item
    int index = get_global_id(0);
    __local float tmp[16777216];
    mergeSort(x, tmp, size);
}

