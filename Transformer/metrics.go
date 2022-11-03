package ml

import (
	"gorgonia.org/gorgonia"
)

type MetricFunc = func(*gorgonia.Node, *gorgonia.Node) (*gorgonia.Node, error)

// Mae returns a node which is the mean absolute error between two nodes.
func Mae(pred, y *gorgonia.Node) (*gorgonia.Node, error) {
	// The math expression is mae = mean(|err|)
	sub, err := gorgonia.Sub(pred, y)
	if err != nil {
		return nil, errs.ErrorCreatingNode
	}

	absError, err := gorgonia.Abs(sub)
	if err != nil {
		return nil, errs.ErrorCreatingNode
	}

	cost, err := gorgonia.Mean(absError)
	if err != nil {
		return nil, errs.ErrorCreatingNode
	}

	return cost, nil
}

// Mse returns a node which is the mean square error between two nodes.
func Mse(pred, y *gorgonia.Node) (*gorgonia.Node, error) {
	// The math expression is mse = mean(err²)
	sub, err := gorgonia.Sub(pred, y)
	if err != nil {
		return nil, errs.ErrorCreatingNode
	}

	squaredError, err := gorgonia.Square(sub)
	if err != nil {
		return nil, errs.ErrorCreatingNode
	}

	cost, err := gorgonia.Mean(squaredError)
	if err != nil {
		return nil, errs.ErrorCreatingNode
	}

	return cost, nil
}

// Rmse returns a node which is the root mean square error between two nodes.
func Rmse(pred, y *gorgonia.Node) (*gorgonia.Node, error) {
	// The math expression is rmse = sqrt(mse)
	mse, err := Mse(pred, y)
	if err != nil {
		return nil, errs.ErrorCreatingNode
	}

	cost, err := gorgonia.Sqrt(mse)
	if err != nil {
		return nil, errs.ErrorCreatingNode
	}

	return cost, nil
}

// R2 returns a node which is the coefficient of determination between two nodes.
func R2(pred, y *gorgonia.Node) (*gorgonia.Node, error) { //nolint:cyclop
	// For the math expression : https://en.wikipedia.org/wiki/Coefficient_of_determination
	sub, err := gorgonia.Sub(y, pred)
	if err != nil {
		return nil, errs.ErrorCreatingNode
	}

	squaredError, err := gorgonia.Square(sub)
	if err != nil {
		return nil, errs.ErrorCreatingNode
	}

	numerator, err := gorgonia.Sum(squaredError)
	if err != nil {
		return nil, errs.ErrorCreatingNode
	}

	mean, err := gorgonia.Mean(y)
	if err != nil {
		return nil, errs.ErrorCreatingNode
	}

	subM, err := gorgonia.Sub(y, mean)
	if err != nil {
		return nil, errs.ErrorCreatingNode
	}

	squaredSubM, err := gorgonia.Square(subM)
	if err != nil {
		return nil, errs.ErrorCreatingNode
	}

	denominator, err := gorgonia.Sum(squaredSubM)
	if err != nil {
		return nil, errs.ErrorCreatingNode
	}

	d, err := gorgonia.Div(numerator, denominator)
	if err != nil {
		return nil, errs.ErrorCreatingNode
	}

	one := gorgonia.NewConstant(float64(1))

	res, err := gorgonia.Sub(one, d)
	if err != nil {
		return nil, errs.ErrorCreatingNode
	}

	return res, nil
}

// Accuracy returns a node which is the ratio of correctly predicted observation to the total observations.
// (TP + TN) / (TP + TN + FP + FN)
func Accuracy(pred, y *gorgonia.Node) (*gorgonia.Node, error) {
	sub, err := gorgonia.Sub(pred, y)
	if err != nil {
		return nil, errs.ErrorCreatingNode
	}

	absError, err := gorgonia.Abs(sub)
	if err != nil {
		return nil, errs.ErrorCreatingNode
	}

	one := gorgonia.NewConstant(float64(1))

	inv, err := gorgonia.Sub(one, absError)
	if err != nil {
		return nil, errs.ErrorCreatingNode
	}

	cost, err := gorgonia.Mean(inv)
	if err != nil {
		return nil, errs.ErrorCreatingNode
	}

	return cost, nil
}

//Recall returns a node which is the ratio of correctly predicted positive observations to the all observations in actual class
// TP / (TP + FN)
func Recall(pred, y *gorgonia.Node) (*gorgonia.Node, error) {
	ltp, err := gorgonia.HadamardProd(pred, y)
	if err != nil {
		return nil, errs.ErrorCreatingNode
	}

	lfn, err := gorgonia.Lt(pred, y, true)
	if err != nil {
		return nil, errs.ErrorCreatingNode
	}

	tp, err := gorgonia.Sum(ltp)
	if err != nil {
		return nil, errs.ErrorCreatingNode
	}

	fn, err := gorgonia.Sum(lfn)
	if err != nil {
		return nil, errs.ErrorCreatingNode
	}

	sum, err := gorgonia.Add(tp, fn)
	if err != nil {
		return nil, errs.ErrorCreatingNode
	}

	res, err := gorgonia.Div(tp, sum)
	if err != nil {
		return nil, errs.ErrorCreatingNode
	}

	return res, nil
}

// Precision returns a node which is the ratio of correctly predicted positive observations to the total predicted positive observations.
// TP / (TP + FP)
func Precision(pred, y *gorgonia.Node) (*gorgonia.Node, error) {
	ltp, err := gorgonia.HadamardProd(pred, y)
	if err != nil {
		return nil, errs.ErrorCreatingNode
	}

	tp, err := gorgonia.Sum(ltp)
	if err != nil {
		return nil, errs.ErrorCreatingNode
	}

	lfp, err := gorgonia.Lt(y, pred, true)
	if err != nil {
		return nil, errs.ErrorCreatingNode
	}

	fp, err := gorgonia.Sum(lfp)
	if err != nil {
		return nil, errs.ErrorCreatingNode
	}

	sum, err := gorgonia.Add(fp, tp)
	if err != nil {
		return nil, errs.ErrorCreatingNode
	}

	res, err := gorgonia.Div(tp, sum)
	if err != nil {
		return nil, errs.ErrorCreatingNode
	}

	return res, nil
}

// F1 returns a node which is the weighted average of Precision and Recall.
// F1Score = 2*(Recall * Precision) / (Recall + Precision)
func F1(pred, y *gorgonia.Node) (*gorgonia.Node, error) {
	//F1 Score = 2*(Recall * Precision) / (Recall + Precision)
	zero5 := gorgonia.NewConstant(0.5)

	ltp, err := gorgonia.HadamardProd(pred, y)
	if err != nil {
		return nil, errs.ErrorCreatingNode
	}

	tp, err := gorgonia.Sum(ltp)
	if err != nil {
		return nil, errs.ErrorCreatingNode
	}

	lfp, err := gorgonia.Lt(y, pred, true)
	if err != nil {
		return nil, errs.ErrorCreatingNode
	}

	lfn, err := gorgonia.Lt(pred, y, true)
	if err != nil {
		return nil, errs.ErrorCreatingNode
	}

	sum, err := gorgonia.Add(lfn, lfp)
	if err != nil {
		return nil, errs.ErrorCreatingNode
	}

	mul, err := gorgonia.Mul(sum, zero5)
	if err != nil {
		return nil, errs.ErrorCreatingNode
	}

	sum2, err := gorgonia.Add(mul, ltp)
	if err != nil {
		return nil, errs.ErrorCreatingNode
	}

	sum3, err := gorgonia.Sum(sum2)
	if err != nil {
		return nil, errs.ErrorCreatingNode
	}

	res, err := gorgonia.Div(tp, sum3)
	if err != nil {
		return nil, errs.ErrorCreatingNode
	}

	return res, nil
}

// LogLoss returns a node which is the Cross-entropy loss between two node
func LogLoss(pred, y *gorgonia.Node) (*gorgonia.Node, error) {
	//Formula : −1/N*SUM(y*log(pred)+(1−y)*log(1−pred))
	one := gorgonia.NewConstant(float64(1))
	minusOne := gorgonia.NewConstant(float64(-1))

	logPred, err := gorgonia.Log(pred)
	if err != nil {
		return nil, errs.ErrorCreatingNode
	}

	minusPred, err := gorgonia.Sub(one, pred)
	if err != nil {
		return nil, errs.ErrorCreatingNode
	}

	logMinusHyp, err := gorgonia.Log(minusPred)
	if err != nil {
		return nil, errs.ErrorCreatingNode
	}

	minusY, err := gorgonia.Sub(one, y)
	if err != nil {
		return nil, errs.ErrorCreatingNode
	}

	mul1, err := gorgonia.HadamardProd(y, logPred)
	if err != nil {
		return nil, errs.ErrorCreatingNode
	}

	mul2, err := gorgonia.HadamardProd(minusY, logMinusHyp)
	if err != nil {
		return nil, errs.ErrorCreatingNode
	}

	add, err := gorgonia.Add(mul2, mul1)
	if err != nil {
		return nil, errs.ErrorCreatingNode
	}

	minusAdd, err := gorgonia.Mul(minusOne, add)
	if err != nil {
		return nil, errs.ErrorCreatingNode
	}

	res, err := gorgonia.Mean(minusAdd)
	if err != nil {
		return nil, errs.ErrorCreatingNode
	}

	return res, nil
}
