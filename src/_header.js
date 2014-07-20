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


    