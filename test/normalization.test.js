"use strict";

var base = require('./_base'),
  sinon = base.sinon,
  assert = base.assert,
  expect = base.expect,
  should = base.should;


var ml;


var test = module.exports = {
  beforeEach: function() {
    ml = require('../index')();
  },
  'normalize features': {
    'returns mean and std': function() {
      var m = new ml.Matrix([ [1, 2, 3], [1, 5, 9], [1, -10, 81] ]);

      var ret = ml.normalizeFeatures(m);

      ret.mean.data.should.eql([ [1, -1, 31] ]);
      base.round(ret.std.data[0]).should.eql([0, 7.9373, 43.4051]);
    },
    'returns new matrix': function() {
      var m = ml.Matrix.identity(3);

      var m2 = ml.normalizeFeatures(m).X;

      m2.should.not.eql(m);
    },
    'prepends x0 column': function() {
      var a = [ [1], [41], [7] ];
      var m = new ml.Matrix(a);

      var m2 = ml.normalizeFeatures(m).X;

      m2.data[0][0].should.eql(1);
      m2.data[1][0].should.eql(1);
      m2.data[2][0].should.eql(1);
    },
    'normalizes the original values by column-by-column': function() {
      var a = [ [1, -2, 3], [41, 5, 6], [7, 80, 9] ];
      var m = new ml.Matrix(a);

      var m2 = ml.normalizeFeatures(m).X;

      m2.data.should.eql([ 
        [ 1, -0.7108115812977249, -0.652632295604179, -1 ],
        [ 1, 1.143479500348514, -0.49864040563015927, 0 ],
        [ 1, -0.43266791905078905, 1.1512727012343382, 1 ] 
      ]);
    },
    'works even if standard deviation of a given column is 0': function() {
      var a = [ [5], [5], [5] ];
      var m = new ml.Matrix(a);

      var m2 = ml.normalizeFeatures(m).X;

      m2.data.should.eql([ [1, 0], [1, 0], [1, 0] ]);
    }
  }
};



