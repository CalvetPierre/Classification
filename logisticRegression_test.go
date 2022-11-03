package predictors_test

import (
	"testing"

	"github.com/go-gota/gota/dataframe"
	"github.com/go-gota/gota/series"

)

// classify returns a list made of 1 and 0 when values in the initial list are above or below mean. duh (:
func classify(list []float64, mean float64) []float64 {
	for i, val := range list {
		if val < mean {
			list[i] = 0.
		} else {
			list[i] = 1.
		}
	}

	return list
}

func TestLogRegEvaluate(t *testing.T) {
	// Import df from CSV
	xDF, yDF, err := ml.ImportTest()
	if err != nil {
		t.Error("Error importing the df: ", err)
	}

	*yDF = yDF.Mutate(series.New(classify(yDF.Col("y").Float(), yDF.Col("y").Mean()), series.Float, "y"))

	lr := predictors.NewLogisticRegression(40000, 0.002, false)

	if err := lr.Fit(xDF, yDF); err != nil {
		t.Error("an error occurred during the fitting of the logistic regression: ", err)
	}

	acc, err := lr.Evaluate(xDF, yDF, ml.Accuracy)
	if err != nil {
		t.Error("an error occurred in Evaluate(Accuracy)")
	}

	recall, err := lr.Evaluate(xDF, yDF, ml.Recall)
	if err != nil {
		t.Error("an error occurred in Evaluate(Recall)")
	}

	precision, err := lr.Evaluate(xDF, yDF, ml.Precision)
	if err != nil {
		t.Error("an error occurred in Evaluate(Precision)")
	}

	f1, err := lr.Evaluate(xDF, yDF, ml.F1)
	if err != nil {
		t.Error("an error occurred in Evaluate(F1)")
	}

	if acc > 1 || acc < 0.9 {
		t.Error("Wrong accuracy value")
		t.Log("expected around 0.9")
		t.Log("got : ", acc)
	}

	if precision > 1 || precision < 0.9 {
		t.Error("Wrong precision value")
		t.Log("expected around 0.9")
		t.Log("got : ", precision)
	}

	if recall > 1 || recall < 0.9 {
		t.Error("Wrong Recall value")
		t.Log("expected around 0.9")
		t.Log("got : ", recall)
	}

	if f1 > 1 || f1 < 0.9 {
		t.Error("Wrong F1 score value")
		t.Log("expected around 0.9")
		t.Log("got : ", f1)
	}
}

func TestLogRegPredictProba(t *testing.T) {
	// Import df from CSV
	xDF, yDF, err := ml.ImportTest()
	if err != nil {
		t.Error("Error importing the df: ", err)
	}

	*yDF = yDF.Mutate(series.New(classify(yDF.Col("y").Float(), yDF.Col("y").Mean()), series.Float, "y"))

	lr := predictors.NewLogisticRegression(100000, 0.002, false)

	scaler := transformers.NewMinMaxScaler([]string{"X"})

	if err := scaler.Fit(xDF); err != nil {
		t.Error("failed Fit", err)
	}
	if err := scaler.Transform(xDF); err != nil {
		t.Error("failed Transform: ", err)
	}

	if err := lr.Fit(xDF, yDF); err != nil {
		t.Error("an error occurred during the fitting of the logistic regression: ", err)
	}

	// Create a new vector and predict its price
	val := dataframe.LoadRecords(
		[][]string{
			{"X"},
			{"0.5"},
		},
	)

	prob, err := lr.PredictProba(&val)
	if err != nil {
		t.Error("an error occurred in PredictProba", err)
	}

	if p := prob.Value().Data().([]float64)[0]; p < 0.45 || p > 0.55 {
		t.Error("Wrong predicted value")
		t.Log("expected around 0.5")
		t.Log("got : ", p)
	}
}
