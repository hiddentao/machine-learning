(function (root, factory) {
  "use strict";

  // AMD
  if (typeof define === 'function' && define.amd) {
    define(['linear-algebra'], factory);
  }
  // CommonJS
  else if (typeof exports === 'object') {
    module.exports = factory(require('linear-algebra'));
  }
  // Browser
  else {
    root.machineLearning = factory(root.linearAlgebra);
  }
})(this, function (defaultLinearAlgebra) {
  "use strict";


  var _SMALLEST_DELTA = Number.EPSILON || 0.000000000001;


  /**
   * Initialise the machine learning library.
   *
   * @param {Object} linearAlgebra Linear algebra library.
   * 
   * @return {Object} Initialized machine learning algorithms.
   */
  return function(linearAlgebra) {
    if (!linearAlgebra) {
      linearAlgebra = defaultLinearAlgebra();
    }

    // namespace
    var ML = {
      Matrix: linearAlgebra.Matrix,
      Vector: linearAlgebra.Vector
    };


    
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



/**
 * Construct a new linear regression solver.
 *
 * @param {Integer} numFeatures The number of features in the training set.
 * 
 * @constructor
 */
ML.LinearReg = function(numFeatures) {
  this._dataX = [];
  this._dataY = [];
  this._nF = numFeatures;   // need extra space to hold x0
};




/**
 * Add data to the training set.
 *
 * Each row in the training set is an Array consisting of the `x` values 
 * (the features) followed by the `y` value (the output) at the end.
 *
 * The no. of features is expected to match the value passed in during 
 * construction.
 * 
 * @param  {Array} newTrainingData One or more rows (Array) of training data.
 * @return this
 */
ML.LinearReg.prototype.addData = function(data) {
  var rows = data.length;

  for (var i=0; i<data.length; ++i) {
    this._dataX.push(data[i].slice(0, rows-1));
    this._dataY.push(data[i][rows-1]);
  }
};




/**
 * Solve this regression using gradient descent.
 *
 * @param {Number} alpha Initial learning rate.
 * @param {Integer} maxIters Max. no. of iterations to perform.
 * 
 * @return {Object} {theta: Vector, cost: float, alpha: float, iters: int}
 */
ML.LinearReg.prototype.solve = function(alpha, maxIters) {
  var X = new ML.Matrix(this._dataX),
    y = new ML.Vector(this._dataY);

  return ML.gradientDescent(X, y, ML.LinearReg.costFunction);
};






/**
 * The cost function.
 * @return {Number}
 */
ML.LinearReg.costFunction = function(X, theta, y) {
  /*
    m = y.size
   
    Cost = 1/2m * trans(XT*theta - y) * (XT*theta - y), where XT=trans(X)
  */
  var X_mul_theta_minus_y = X.mul(theta).minusP(y);

  return (X_mul_theta_minus_y.dot(X_mul_theta_minus_y)) / (2 * y.size);
};







    return ML;
  };
});

