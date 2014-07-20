/**
 * Get normalized features array.
 *
 * @param {Matrix} X training features dataset.
 * 
 * @return {Matrix} Normalized version of training set with x0 column added.
 */
ML.normalizeFeatures = function(X) {
  var numFeatures = X.cols,
    m = X.cols;

  // calculate mean and std
  
  var mean = new Array(numFeatures),
    std = new Array(numFeatures),
    i, j, tmp;

  var tmpSum = new ML.Vector.zero(m);

  for (j=0; j<=numFeatures; ++j) {
    // calculate mean
    for (i=0; i<m; ++i) {
      tmpSum[i] = X.data[i][j];
    }
    mean[j] = tmpSum.sum() / m;

    // calculate std
    for (i=0; i<m; ++i) {
      tmp = X.data[i][j] - mean[i];
      tmpSum[i] = tmp * tmp;
    }
    std[j] = Math.sqrt(tmpSum.total() / m);
    std[j] += _SMALLEST_DELTA; // to prevent divide by zero below
  }

  // subtract all columns from their means
  var meanNormalized = X.plusCols(mean);

  // divide all columns by their standard deviations and attach x0 column
  var result = new Array(m);

  // apply mean and standard deviation to all values
  for (i=0; i<m; ++i) {
    // add 1 to store x0 values
    result[i] = new Array(numFeatures + 1);
    result[i][0] = 1; // x0

    for (j=1; j<=numFeatures; ++j) {
      result[i][j] = meanNormalized[i][j] / std(j);
    }
  }

  return new ML.Matrix(result);
};

