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


  var _throwError = function(cat, op, msg) {
    throw new Error('machine-learning: [' + cat + '_' + op + '] ' + msg);
  };


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

  var oldCost, i, j, row, tmpSum = new ML.Vector.zero(m);

  // initial theta and theta delta as column vectors
  var theta = ML.Vector.zero(n).trans_(),
      delta = ML.Vector.zero(n).trans_();

  // initial cost
  var cost = costFn(X, theta, y);

  // loop
  var iters = maxIters;
  while (0 < cost && 0 < iters--) {
    // save current cost
    oldCost = cost;

    // for each theta value
    for (j=0; j<n; ++j) {
      // to calculate derivate we go through each row in training set
      for (i=0; i<m; ++i) {
        tmpSum[i] = (X.dot(i, theta) - y[i]) * X.data[i][j];
      }

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



/**
 * Construct a new linear regression solver.
 *
 * @param {Integer} numFeatures The number of features in the training set.
 * 
 * @constructor
 */
ML.LinReg = function(numFeatures) {
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
ML.LinReg.prototype.addData = function(data) {
  var rows = data.length;

  for (var i=0; i<rows; ++i) {
    if (data[i].length < this._nF) {
      _throwError('LinReg', 'addData', 'Not enough data');
    }

    this._dataX.push(data[i].slice(0, this._nF-1));
    this._dataY.push(data[i][this._nF-1]);
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
ML.LinReg.prototype.solve = function(alpha, maxIters) {
  var X = new ML.Matrix(this._dataX),
    y = new ML.Matrix(this._dataY).trans_();

  return ML.gradientDescent(X, y, ML.LinReg.costFunction);
};






/**
 * The cost function.
 *
 * @param {Matrix} X Size m x n
 * @param {Matrix} theta Size n x 1
 * @param {Matrix} y Size m x 1
 * 
 * @return {Number} Cost using theta
 */
ML.LinReg.costFunction = function(X, theta, y) {
  /*
    Cost = (X*theta - y)^2 * (m / 2)
  */
  var m = y.rows;

  var tmp = X.dot(theta).minus_(y);

  return (tmp.mul(tmp)).getSum() / (2 * m);
};







    return ML;
  };
});

