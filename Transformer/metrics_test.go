package ml_test

import (
	ml "awesomeProject/Transformer"
	"math"
	"testing"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func testMetric(predList, yList []float64, metricFunc ml.MetricFunc) (float64, error) {
	g := gorgonia.NewGraph()
	y := gorgonia.NodeFromAny(g,
		tensor.New(tensor.WithBacking(yList)),
		gorgonia.WithName("y"))
	pred := gorgonia.NodeFromAny(g,
		tensor.New(tensor.WithBacking(predList)),
		gorgonia.WithName("pred"))

	m, err := metricFunc(pred, y)
	if err != nil {
		return 0, errs.ErrorCreatingNode
	}

	machine := gorgonia.NewTapeMachine(g)
	defer machine.Close()

	if err := machine.RunAll(); err != nil {
		return 0, errs.ErrorRunningVM
	}

	return m.Value().Data().(float64), nil
}

func TestMae(t *testing.T) {
	prediction, err := testMetric([]float64{5.0, 6.0, 7.0, 9.0, 8.0}, []float64{1.0, 2.0, 3.0, 4.0, 5.0}, ml.Mae)

	if err != nil {
		t.Error("error running the VM")
	}

	if prediction != 4 {
		t.Error("wrong value calculated")
		t.Log("Found   : ", prediction)
		t.Log("Expected: ", 4)
	}
}

func TestMse(t *testing.T) {
	prediction, err := testMetric([]float64{5.0, 6.0, 7.0, 9.0, 8.0}, []float64{1.0, 2.0, 3.0, 4.0, 5.0}, ml.Mse)

	if err != nil {
		t.Error("error running the VM")
	}

	if prediction != 16.4 {
		t.Error("wrong value calculated")
		t.Log("Found   : ", prediction)
		t.Log("Expected: ", 16.4)
	}
}

func TestRmse(t *testing.T) {
	prediction, err := testMetric([]float64{5.0, 6.0, 7.0, 9.0, 8.0}, []float64{1.0, 2.0, 3.0, 4.0, 5.0}, ml.Rmse)

	if err != nil {
		t.Error("error running the VM")
	}

	if prediction != math.Sqrt(16.4) {
		t.Error("wrong value calculated")
		t.Log("Found   : ", prediction)
		t.Log("Expected: ", math.Sqrt(16.4))
	}
}

func TestR2A(t *testing.T) {
	prediction, err := testMetric([]float64{3.0, 3.0, 3.0, 3.0, 3.0}, []float64{1.0, 2.0, 3.0, 4.0, 5.0}, ml.R2)

	if err != nil {
		t.Error("error running the VM")
	}

	if prediction != 0 {
		t.Error("wrong value calculated")
		t.Log("Found   : ", prediction)
		t.Log("Expected: ", 0)
	}
}

func TestR2B(t *testing.T) {
	prediction, err := testMetric([]float64{1.1, 2.1, 3.1, 4.1, 5.1}, []float64{1.0, 2.0, 3.0, 4.0, 5.0}, ml.R2)

	if err != nil {
		t.Error("error running the VM")
	}

	if prediction < 0.9 {
		t.Error("wrong value calculated")
		t.Log("Found   : ", prediction)
		t.Log("Expected: ", 0)
	}
}

func TestAccuracy(t *testing.T) {
	// 7 TP, 3 FN, 6 TN, 4 FP
	prediction, err := testMetric([]float64{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
		[]float64{1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1}, ml.Accuracy)

	if err != nil {
		t.Error("error running the VM")
	}

	if prediction != 0.65 {
		t.Error("wrong value calculated")
		t.Log("Found   : ", prediction)
		t.Log("Expected: ", 0.65)
	}
}

func TestRecall(t *testing.T) {
	prediction, err := testMetric([]float64{1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1},
		[]float64{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, ml.Recall)

	if err != nil {
		t.Error("error running the VM")
	}

	if prediction != 0.7 {
		t.Error("wrong value calculated")
		t.Log("Found   : ", prediction)
		t.Log("Expected: ", 0.7)
	}
}

func TestPrecision(t *testing.T) {
	prediction, err := testMetric([]float64{1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1},
		[]float64{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, ml.Precision)

	if err != nil {
		t.Error("error running the VM")
	}

	if prediction != 7./11. {
		t.Error("wrong value calculated")
		t.Log("Found   : ", prediction)
		t.Log("Expected: ", 7./11.)
	}
}

func TestF1(t *testing.T) {
	prediction, err := testMetric([]float64{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
		[]float64{1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1}, ml.F1)
	if err != nil {
		t.Error("error running the VM")
	}

	if prediction != 2./3. {
		t.Error("wrong value calculated")
		t.Log("Found   : ", prediction)
		t.Log("Expected: ", 2./3.)
	}
}

func TestLogLoss(t *testing.T) {
	prediction, err := testMetric([]float64{0.9, 0.8, 0.05}, []float64{1.0, 1.0, 0.0}, ml.LogLoss)
	if err != nil {
		t.Error("error running the VM")
	}

	if prediction != -(math.Log(0.9)+math.Log(0.8)+math.Log(0.95))/3 {
		t.Error("wrong value calculated")
		t.Log("Found   : ", prediction)
		t.Log("Expected: ", -(math.Log(0.9)+math.Log(0.8)+math.Log(0.95))/3)
	}
}
