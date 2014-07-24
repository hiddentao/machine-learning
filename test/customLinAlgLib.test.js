"use strict";

var base = require('./_base'),
  sinon = base.sinon,
  assert = base.assert,
  expect = base.expect,
  should = base.should;


var test = module.exports = {
  'custom': function() {
    var custom = {
      Matrix: 1,
      Vector: 2
    };

    var lib = require('../index')(custom);

    lib.Matrix.should.eql(1);
    lib.Vector.should.eql(2);
  }
};
