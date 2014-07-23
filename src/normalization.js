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
  
  var mean, std, i, j, tmpSum = new ML.Vector.zero(m);

  for (j=0; j<=numFeatures; ++j) {
    // calculate mean
    for (i=0; i<m; ++i) {
      tmpSum.data[0][i] = X.data[i][j];

      // pre-allocate final array with space for x0 column
      if (!Array.isArray(newArray[i])) {
        newArray[i] = new Array(numFeatures + 1);
        newArray[i][0] = 1; // x0 column
      }
    }
    mean = tmpSum.getSum() / m;

    // calculate std
    for (i=1; i<=m; ++i) {
      // subtract each column value from the mean
      newArray[i][j] = X.data[i-1][j] - mean;
      // and calculate std at the same time
      tmpSum.data[0][i] = newArray[i][j] * newArray[i][j];
    }
    std = Math.sqrt(tmpSum.getSum() / m);
    std += _SMALLEST_DELTA; // to prevent divide by zero below

    // divide column by std
    for (i=1; i<=m; ++i) {
      // subtract each column value from the mean
      newArray[i][j] /= std;
    }
  }

  return new ML.Matrix(newArray);
};

