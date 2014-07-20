/**
 * Perform gradient descent to compute the value of theta.
 *
 * Once complete this returns the final `theta`, `cost`, the final learning 
 * rate (`alpha`) as well as the number of iterations elpased (`iter`).
 *
 * @param {Matrix} X Training feature data (without x0 column)
 * @param {Vector} y Training result data
 * @param {Function} costFn Function to compute cost.
 * @param {Number} alpha Initial learning rate.
 * @param {Integer} maxIters Max. no. of iterations to perform.
 * 
 * @return {Object} {theta: Vector, cost: float, alpha: float, iters: int}
 */
ML.gradientDescent = function(X, y, costFn, alpha, maxIters) {
  /*
    1. Feature-normalization
    2. Initialise theta to be a zero-vector
    3. For each iteration:    
      3.0 If cost is 0 or maxIters reached then goto 4
      3.1 Calculate new theta, where:
        theta(j) = theta(j) - (alpha / m) * {i=1..m: (X(i)*theta - y(i))*(X(i)(j)) }
      3.2 Update theta
      3.3 Calculate new cost
      3.4 If cost increased then alpha = alpha / 2
    4. Exit and return theta, cost, alpha, iters
  */
  
  // the features array
  X = ML.normalizeFeatures(X);

  // measurements
  var m = X.rows,
    n = X.cols;

  var oldCost, i, j, row, tmp;

  // initial theta and theta delta
  var theta = ML.Vector.zero(n),
      delta = new ML.Vector.zero(n);

  // initial cost
  var cost = costFn(X, theta, y);

  // loop
  var iters = maxIters;
  while (0 < cost && 0 < iters--) {
    // save current cost
    oldCost = cost;

    // for each theta value
    for (j=0; j<n; ++j) {
      tmp = 0;

      // to calculate derivate we go through each row in training set
      for (i=0; i<m; ++i) {
        tmp += (X.dot(i, theta) - y[i]) * X.data[i][j];
      }

      delta.data[j] = alpha * tmp / m;
    }

    // update theta
    theta.minusP(delta);

    // calculate new cost
    cost = costFn(X, theta, y);

    // if cost increased
    if (cost > oldCost) {
      // restore theta
      theta.plusP(delta);
      // reduce alpha
      alpha /= 2;
    }
  }

  return {
    theta: theta,
    cost: cost,
    alpha: alpha,
    iters: maxIters - iters
  };
};


