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
    root.LinearReg = factory(root.linearAlgebra);
  }
})(this, function (linearAlgebra) {
  "use strict";



  /**
   * Construct a new linear regression solver.
   *
   * @param {Integer} numFeatures The number of features in the training set.
   * @param {Object} options Additional options.
   * @param {Function} [options.add] Floating point adder (for higher precision linear algebra).
   * 
   * @constructor
   */
  var LinearReg = function(numFeatures, options) {
    // initialise linear algebra
    this.options = options || {};
    var LinAlg = linearAlgebra({
      add: this.options.add
    });
    this.Vector = LinAlg.Vector;
    this.Matrix = LinAlg.Matrix;

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
  LinearReg.prototype.addData = function(data) {
    var rows = data.length;

    for (var i=0; i<data.length; ++i) {
      this._dataX.push(data[i].slice(0, this._nf-1));
      this._dataY.push(data[i][this._nf-1]);
    }
  };




  /**
   * Get normalized features array.
   * 
   * @return {Matrix} Normalized version of training set with x0 column added.
   */
  LinearReg.prototype._normalizedFeatures = function() {
    var mean = new Array(this._nF),
      std = new Array(this._nF),
      i, j;

    var m = this._dataY.length;

    // calculate mean and standard deviation
    for (j=0; j<=this._nF; ++j) {
      var toAdd = new Array(m);

      for (i=0; i<m; ++i) {
        toAdd[i] = this._dataX[i][j];
      }

      var total = this.options.add(toAdd);
      mean[j] = total / m;
      std[j] = total / m; // TODO
    }

    var result = new Array(m);

    // apply mean and standard deviation to all values
    for (i=0; i<m; ++i) {
      // add 1 to store x0 values
      result[i] = new Array(this._nF + 1);
      result[i][0] = 1; // x0

      for (var j=0; j<this._nf; ++j) {
        result[i][j+1] = this.options.add([ this.dataX[i,j], -mean[j] ]) / std(j);
      }
    }

    return new Matrix(result);
  };




  /**
   * Perform gradient descent to compute the value of theta.
   *
   * Once complete this returns the final `theta`, `cost`, the final learning 
   * rate (`alpha`) as well as the number of iterations elpased (`iter`).
   * 
   * @return {Object} {theta: Vector, cost: float, alpha: float, iters: int}
   */
  LinearReg.prototype.gradientDescent = function(alpha, numIters) {
    /*
      1. Feature-normalization
      2. Initialise theta to be a zero-vector
      3. For each iteration:    
        3.0 If cost is 0 or numIters reached then goto 4
        3.1 Calculate new theta, where:
          theta(j) = theta(j) - (alpha / m) * {i=1..m: (X(i)*theta - y(i))*(X(i)(j)) }
        3.2 Update theta
        3.3 Calculate new cost
        3.4 If cost increased then alpha = alpha / 2
      4. Exit and return theta, cost, alpha, iters
    */
    
    // the features array
    var X = this._normalizedFeatures();

    // the results array
    var y = new this.Vector(this._dataY);

    // measurements
    var m = X.dim[1],
      n = X.dim[2];

    var oldCost, delta, i, j, row, tmp;

    // initial theta
    var theta = this.Vector.zero(n);

    // initial cost
    cost = this._computeCost(X, theta, y);

    // loop
    var X_data = X.data();
    var iters = numIters;
    while (0 < cost && 0 < iters--) {
      // save current cost
      oldCost = cost;

      // theta delta
      delta = this.Vector.zero(n);

      // for each theta value
      for (j=0; j<n; ++j) {
        tmp = new Array(m);

        // to calculate derivate we go through each row in training set
        for (i=0; i<m; ++i) {
          t[i] = (X.dot(i, theta) - y[i]) * X_data[i][j];
        }

        delta[j] = alpha * this.options.add(tmp) / m;
      }

      // update theta
      theta.minusP(delta);

      // calculate new cost
      cost = this._computeCost(X, theta, y);

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
      iters: numIters - iters
    };
  };




  /**
   * Compute the cost given the current value of theta.
   * @return {Number}
   */
  LinearReg.prototype._computeCost = function(X, theta, y) {
    /*
      Cost = 1/2m * trans(XT*theta - y) * (XT*theta - y), where XT=trans(X)
    */
    var m = y.dim;

    var X_mul_theta_minus_y = X.mul(theta).minusP(y);

    return (X_mul_theta_minus_y.dot(X_mul_theta_minus_y)) / (2 * m);
  };



  return LinearReg;
});