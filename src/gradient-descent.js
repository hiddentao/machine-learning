/**
 * Perform gradient descent to compute the value of theta.
 *
 * Once complete this returns the final `theta`, `cost`, the final learning 
 * rate (`alpha`) as well as the number of iterations elpased (`iter`).
 *
 * @param {Matrix} X Training feature data size m x n (without x0 column)
 * @param {Vector} y Training result data, size m x 1
 * @param {Function} costFn Function to compute cost.
 * @param {Number} alpha Initial learning rate.
 * @param {Integer} maxIters Max. no. of iterations to perform.
 * 
 * @return {Object} {theta: Matrix(n+1 x 1), cost: float, alpha: float, iters: int}
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

  var oldCost, i, j, row, h_x, tmpSum = new ML.Vector.zero(m);

  // initial theta and theta delta as column vectors
  var theta = ML.Vector.zero(n).trans_(),
      delta = ML.Vector.zero(n).trans_();

  // initial cost
  var cost = costFn(X, theta, y);

  // loop
  var iters = 0;
  while (0 < cost && maxIters > iters) {
    iters++;

    // save current cost
    oldCost = cost;

    // calculate h(x) - y
    h_x = X.dot(theta).minus_(y);

    // for each theta value
    for (j=0; j<n; ++j) {
      // calculate sum = (h(x(i)) − y(i))*x(i,j)
      for (i=0; i<m; ++i) {
        tmpSum[i] = h_x.data[i][j] * X.data[i][j];
      }
      
      // delta = alpha * (sum / m)
      delta.data[j][0] = alpha * tmpSum.getSum() / m;
    }

    // update theta
    theta.minus_(delta);

    // calculate new cost
    cost = costFn(X, theta, y);

    // if cost increased
    if (cost > oldCost) {
      // restore theta
      theta.plus_(delta);
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


