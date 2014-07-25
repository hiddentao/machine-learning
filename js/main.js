var MINI = require('minified');
var _=MINI._, $=MINI.$, $$=MINI.$$, EE=MINI.EE, HTML=MINI.HTML;


$.ready(function() {

  demo.setupLinearRegression();

});



var demo = {
  /* Common stuff */

  defaultPointOptions: {
    withLabel: false,
    fixed: true,
    strokeOpacity: 0
  },

  getMouseCoords : function(board, e, i) {
      var cPos = board.getCoordsTopLeftCorner(e, i),
          absPos = JXG.getPosition(e, i),
          dx = absPos[0]-cPos[0],
          dy = absPos[1]-cPos[1];

      return new JXG.Coords(JXG.COORDS_BY_SCREEN, [dx, dy], board);
  },
  buildAddMousePointToBoardHandler: function(board, pointOptions, cb) {
    if (2 === arguments.length) {
      cb = pointOptions;
      pointOptions = {};
    }

    return function(e) {
      var canCreate = true, i, coords, el;
      
      if (e[JXG.touchProperty]) {
          // index of the finger that is used to extract the coordinates
          i = 0;
      }
      coords = demo.getMouseCoords(board, e, i);
      
      for (el in board.objects) {
          if(JXG.isPoint(board.objects[el]) && board.objects[el].hasPoint(coords.scrCoords[1], coords.scrCoords[2])) {
              canCreate = false;
              break;
          }
      }
      
      if (canCreate) {
        var x = coords.usrCoords[1], y = coords.usrCoords[2];

        board.create('point', [x, y], 
          _.extend({}, demo.defaultPointOptions, pointOptions)
        );

        if (cb) {
          cb(x, y)
        }
      }
    };
  },
  createBoard : function(elementId, options) {
    options = _.extend({
      showCopyright: false,
      showNavigation: false,
      zoom: false,
      pan: false
    }, options);

    return JXG.JSXGraph.initBoard(elementId, options);
  },
  createHiddenPoint : function(board, coords) {
    return board.create('point', coords, _.extend({}, demo.defaultPointOptions, { 
      visible: false, 
    }));
  },
  webWorker: function(fn) {
    return cw({
      init: function() {
        importScripts("js/linear-algebra.min.js");
        importScripts("js/machine-learning.min.js");
        this.ML = self.machineLearning();
      },
      run: fn
    });
  },

  /* Linear Regression */

  setupLinearRegression: function() {
    var points = [],
      worker = null;

    // create board
    board = demo.createBoard('linreg-canvas', {
      boundingbox: [-2,10,10,-2], 
      axis: true,
    });

    // create best-fit line
    var linePoints = [
      demo.createHiddenPoint(board ,[0,0]),
      demo.createHiddenPoint(board ,[0,0])
    ];
    board.create('line', linePoints, { 
      fillColor: '#212f59' 
    });

    // click to add point
    board.on('down', demo.buildAddMousePointToBoardHandler(board, function recalculate(x, y) {
      // update points
      points.push([x, y]);

      // if not enough points then skip
      if (2 > points.length) {
        return;
      }

      // kill existing worker
      if (worker) {
        worker.close();
      }

      // create new worker
      worker = demo.webWorker(function(points) {
        var linreg = new this.ML.LinReg(1);
        linreg.addData(points);
        
        var ret = linreg.solve(0.01, 1000);

        return {
          theta: ret.theta.toArray(),
          mean: ret.mean.toArray(),
          std: ret.std.toArray()
        }
      });

      // run it!
      worker.run(points).then(function(result) {
        var mean = $(result.mean),
          std = $(result.std),
          theta = $(result.theta);

        // calculate two points to make the line: y = t0 + t1*((x-m)/std)
        var x = -1;
        linePoints.forEach(function(p) {
          x += 1;
          var y = theta[0] + theta[1] * ((x-mean[0]) / std[0]);
          p.moveTo([x, y], 500);
        });
      });
    }));

  },

};



