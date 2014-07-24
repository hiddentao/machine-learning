"use strict";

var sinon = require('sinon'),
  chai = require('chai'),
  expect = chai.expect,
  should = chai.should();
chai.use(require('sinon-chai'));


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
