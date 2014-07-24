/**
 * Get normalized features array.
 *
 * @param {Matrix} X training features dataset.
 * 
 * @return {Matrix} Normalized version of training set with x0 column added.
 */
ML.normalizeFeatures = function(X) {
  var numFeatures = X.cols,
    m = X.rows;

  // for each value in each column, subtract column mean from it 
  // and divide by result by column std (standard deviation)

  var newArray = new Array(m);
  
  var mean, std, i, j, jNormalized, tmpSum = new ML.Vector.zero(m);

  for (j=0; j<numFeatures; ++j) {
    jNormalized = j + 1;  // because we'll be prepending an x0 column later on

    // calculate mean
    for (i=0; i<m; ++i) {
      tmpSum.data[0][i] = X.data[i][j];

      // pre-allocate final array with space for x0 column
      if (!Array.isArray(newArray[i])) {
        newArray[i] = new Array(numFeatures + 1); // with space for x0 column
        newArray[i][0] = 1; // x0 column
      }
    }
    mean = tmpSum.getSum() / m;

    // calculate std
    for (i=0; i<m; ++i) {
      // subtract each column value from the mean
      newArray[i][jNormalized] = X.data[i][j] - mean;
      // and calculate std at the same time
      tmpSum.data[0][i] = newArray[i][jNormalized] * newArray[i][jNormalized];
    }
    std = Math.sqrt(tmpSum.getSum() / (m-1));   // m-1 for unbiased estimation (Bessel's correction)
    if (0 === std) { 
      std = _SMALLEST_DELTA; // to prevent divide by zero below
    }

    // divide column by std
    for (i=0; i<m; ++i) {
      // subtract each column value from the mean
      newArray[i][jNormalized] /= std;
    }
  }

  return new ML.Matrix(newArray);
};


