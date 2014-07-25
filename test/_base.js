var fs = require('fs'),
  path = require('path');


var sinon = exports.sinon = require('sinon'),
  chai = require('chai'),
  assert = exports.assert = chai.assert,
  expect = exports.expect = chai.expect,
  should = exports.should = chai.should();
chai.use(require('sinon-chai'));


exports.loadLinRegData = function(filename) {
  var str = fs.readFileSync(path.join(__dirname, 'data', filename)).toString();

  var lines = str.split("\n");

  var features = [], 
    results = [];

  lines.forEach(function(l) {
    var l = l.trim();

    if (l.length) {
      var num = l.split(',').map(function(el) {
        return parseFloat(el);
      });

      features.push(num.slice(0, num.length-1));
      results.push([ num[num.length-1] ]);
    }
  });

  return {
    X: features,
    y: results
  };
};


exports.round = function(arr) {
  return arr.map(function(v) {
    return parseFloat(v.toFixed(4));
  });
};