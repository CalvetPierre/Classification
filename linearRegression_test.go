/*
 * Ratel, a 3D file checker, analyzer and renderer built by Marklix.
 * Copyright (c) 2019-2021, Marklix SAS. All rights reserved for all countries.
 */

package predictors_test

import (
	"testing"

	"github.com/go-gota/gota/dataframe"

)

func TestLinRegPredict(t *testing.T) {
	// Import df from CSV
	xDF, yDF, err := ml.ImportTest()
	if err != nil {
		t.Error("Error importing the df: ", err)
	}

	// Create and fit LinReg
	lr := predictors.NewLinearRegression(ml.Mse, 4000, 0.002, false)

	if err := lr.Fit(xDF, yDF); err != nil {
		t.Error("an error occurred during the fitting of the linear regression: ", err)
	}

	// Create a new df and predict its price
	val := dataframe.LoadRecords(
		[][]string{
			{"X"},
			{"15"},
		},
	)

	pred, err := lr.Predict(&val)
	if err != nil || pred == nil {
		t.Error("an error occurred during the prediction using the linear regression: ", err)
	}

	if prediction := pred.Value().Data().([]float64)[0]; prediction < 14 || prediction > 16 {
		t.Error("wrong predicted value")
		t.Log("Found          : ", prediction)
		t.Log("Expected around: ", 15)
	}
}

func TestLinRegEvaluate(t *testing.T) {
	// Import df from CSV
	xDF, yDF, err := ml.ImportTest()
	if err != nil {
		t.Error("Error importing the df: ", err)
	}

	// Create and fit LinReg
	lr := predictors.NewLinearRegression(ml.Mse, 4000, 0.002, false)

	if err := lr.Fit(xDF, yDF); err != nil {
		t.Error("an error occurred during the fitting of the linear regression")
	}
	// Evaluate the regression on the training dataset
	m, err := lr.Evaluate(xDF, yDF, ml.R2)
	if err != nil {
		t.Error("an error occurred during the fitting of the linear regression")
	}

	if m < 0.9 {
		t.Error("wrong determination coefficient R2")
		t.Log("Found          : ", m)
		t.Log("Expected around: ", 0.9)
	}
}

func TestSaveAndLoad(t *testing.T) {
	// Import df from CSV
	xDF, yDF, err := ml.ImportTest()
	if err != nil {
		t.Error("Error importing the df: ", err)
	}

	// Create and fit LinReg
	lr := predictors.NewLinearRegression(ml.Mse, 4000, 0.002, false)

	if err := lr.Fit(xDF, yDF); err != nil {
		t.Error("an error occurred during the fitting of the linear regression")
	}

	if err := lr.SaveWeights("theta.bin"); err != nil {
		t.Error("error saving weights")
	}
	// Create a new linReg and load the Weights
	lr2 := predictors.NewLinearRegression(ml.Mse, 0, 0.0, false)
	if err := lr2.LoadWeights("theta.bin"); err != nil {
		t.Error("error loading weights")
	}
	// Create a new vector and predict its price
	val := dataframe.LoadRecords(
		[][]string{
			{""},
			{"15"},
		},
	)

	pred, err := lr.Predict(&val)
	if err != nil || pred == nil {
		t.Error("an error occurred during the prediction using the linear regression")
	}

	if prediction := pred.Value().Data().([]float64)[0]; prediction < 14 || prediction > 16 {
		t.Error("wrong predicted value")
		t.Log("Found          : ", prediction)
		t.Log("Expected around: ", 15)
	}
}

func TestPredictUnfitted(t *testing.T) {
	// Create and fit LinReg
	lr := predictors.NewLinearRegression(ml.Mse, 4000, 0.002, false)
	// Create a new df and predict its price
	val := dataframe.LoadRecords(
		[][]string{
			{"X"},
			{"15"},
		},
	)

	if _, err := lr.Predict(&val); err == nil {
		t.Error("not raising an error when trying to predict an unfitted linReg")
	}
}
