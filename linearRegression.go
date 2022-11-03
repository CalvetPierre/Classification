/*
 * Ratel, a 3D file checker, analyzer and renderer built by Marklix.
 * Copyright (c) 2019-2021, Marklix SAS. All rights reserved for all countries.
 */

package predictors

import (
	"encoding/gob"
	"github.com/go-gota/gota/dataframe"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
	"log"
	"os"
)

// LinearRegression contains a gorgonia graph which will be use for the regression.
type LinearRegression struct {
	g            *gorgonia.ExprGraph
	Theta, cost  *gorgonia.Node
	loss         ml.MetricFunc
	iter         int
	learningRate float64
	verbose      bool
}

// NewLinearRegression initializes a LinearRegression.
func NewLinearRegression(loss ml.MetricFunc, iter int, learningRate float64, verbose bool) LinearRegression {
	g := gorgonia.NewGraph()

	return LinearRegression{g: g, loss: loss, iter: iter, learningRate: learningRate, verbose: verbose}
}

// createGraph creates the equation graph used to Fit the model.
func (lr *LinearRegression) createGraph(xT, yT *tensor.Dense, loss ml.MetricFunc) error {
	if xT == nil || yT == nil {
		return errs.ErrorNilPointer
	}

	// Initialize a graph
	lr.g = gorgonia.NewGraph()
	// Create the nodes X, y and theta
	x := gorgonia.NodeFromAny(lr.g, xT, gorgonia.WithName("x"))
	y := gorgonia.NodeFromAny(lr.g, yT, gorgonia.WithName("y"))
	lr.Theta = gorgonia.NewVector(
		lr.g,
		gorgonia.Float64,
		gorgonia.WithName("Theta"),
		gorgonia.WithShape(xT.Shape()[1]),
		gorgonia.WithInit(gorgonia.Uniform(0, 1)))

	// Link the nodes according to the regression equation : Theta * X = pred
	pred, err := gorgonia.Mul(x, lr.Theta)
	if err != nil {
		return errs.ErrorCreatingNode
	}

	// Link the prediction and the real value with the cost equation
	lr.cost, err = loss(pred, y)
	if err != nil {
		return errs.ErrorCreatingNode
	}

	// We want to minimize the cost between pred and y to have Theta * X the closest from y
	if _, err := gorgonia.Grad(lr.cost, lr.Theta); err != nil {
		return errs.ErrorCreatingNode
	}

	return nil
}

// Fit creates a VirtualMachine to run the graph and optimise theta.
func (lr *LinearRegression) Fit(xTrain, yTrain *dataframe.DataFrame) error { //nolint:cyclop
	if xTrain == nil || yTrain == nil {
		return errs.ErrorNilPointer
	}
	// Add a column of ones named bias for the intercept coefficient
	ml.AddBias(xTrain)
	// Transform df to mat
	xT, err := ml.DfToMat(xTrain)
	if err != nil {
		return err
	}

	yT, err := ml.DfToMat(yTrain)
	if err != nil {
		return err
	}

	s := yT.Shape()
	if err := yT.Reshape(s[0]); err != nil {
		return errs.ErrorReshaping
	}

	// Create the equation graph
	if err := lr.createGraph(xT, yT, lr.loss); err != nil {
		return err
	}

	// Defining the VM
	machine := gorgonia.NewTapeMachine(lr.g, gorgonia.BindDualValues(lr.Theta))
	model := []gorgonia.ValueGrad{lr.Theta}
	solver := gorgonia.NewVanillaSolver(gorgonia.WithLearnRate(lr.learningRate))

	// Iterations of the VM
	for i := 0; i < lr.iter; i++ {
		if err := machine.RunAll(); err != nil {
			return errs.ErrorRunningVM
		}

		if err := solver.Step(model); err != nil {
			return errs.ErrorRunningVM
		}

		if lr.verbose && (i%(lr.iter/10.0) == 0) { //nolint:gomnd
			log.Print("Theta:", lr.Theta.Value(), " Iter:", i, " Cost:", lr.cost.Value())
		}

		machine.Reset() // Reset is necessary in a loop like this
	}

	if err := machine.Close(); err != nil {
		return errs.ErrorRunningVM
	}

	return nil
}

// Predict uses the weights in theta to create the target matrix from the given matrix.
func (lr *LinearRegression) Predict(df *dataframe.DataFrame) (*gorgonia.Node, error) {
	// Check if lr is fitted
	if lr.Theta == nil {
		return nil, errs.ErrorUnfitted
	}

	// Add the bias column
	ml.AddBias(df)

	// Transform df to mat
	t, err := ml.DfToMat(df)
	if err != nil {
		return nil, err
	}

	// Create the equation graph
	g := gorgonia.NewGraph()
	theta := gorgonia.NodeFromAny(g, lr.Theta.Value(), gorgonia.WithName("Theta"))
	x := gorgonia.NodeFromAny(g, t, gorgonia.WithName("x"))

	y, err := gorgonia.Mul(x, theta)
	if err != nil {
		return nil, errs.ErrorCreatingNode
	}

	// Create and run the VM
	machine := gorgonia.NewTapeMachine(g)
	if err := machine.RunAll(); err != nil {
		return nil, errs.ErrorRunningVM
	}

	if err := machine.Close(); err != nil {
		return nil, errs.ErrorRunningVM
	}

	return y, nil
}

// Evaluate returns the metric between yT and the prediction from xT.
func (lr *LinearRegression) Evaluate(xTest, yTest *dataframe.DataFrame, metric ml.MetricFunc) (float64, error) {
	// Check if lr is fitted
	if lr.Theta == nil {
		return 0.0, errs.ErrorUnfitted
	}

	// Transform df to mat
	yT, err := ml.DfToMat(yTest)
	if err != nil {
		return 0.0, err
	}

	if err := yT.Reshape(yT.Shape()[0]); err != nil {
		return 0.0, errs.ErrorReshaping
	}

	// Compute the prediction from xTest
	prediction, err := lr.Predict(xTest)
	if err != nil {
		return 0, err
	}

	return GenericEvaluate(prediction, yT, metric)
}

// SaveWeights saves theta's weights in fileName (required type : .bin).
func (lr *LinearRegression) SaveWeights(fileName string) error {
	// Check if lr is fitted
	if lr.Theta == nil {
		return errs.ErrorUnfitted
	}
	// SaveWeights saves the weights of the regression as a bin file
	f, err := os.Create(fileName)
	if err != nil {
		return errs.ErrorEncoder
	}

	enc := gob.NewEncoder(f)

	if err := enc.Encode(lr.Theta.Value()); err != nil {
		return errs.ErrorEncoder
	}

	if err := f.Close(); err != nil {
		return errs.ErrorEncoder
	}

	return nil
}

// LoadWeights loads the weights saved in fileName into lr.
func (lr *LinearRegression) LoadWeights(fileName string) error {
	f, err := os.Open(fileName)
	if err != nil {
		return errs.ErrorEncoder
	}

	dec := gob.NewDecoder(f)

	var thetaT *tensor.Dense

	err = dec.Decode(&thetaT)
	if err != nil {
		return errs.ErrorEncoder
	}

	if err := f.Close(); err != nil {
		return errs.ErrorEncoder
	}

	theta := gorgonia.NodeFromAny(lr.g, thetaT, gorgonia.WithName("Theta"))
	lr.Theta = theta

	return nil
}
