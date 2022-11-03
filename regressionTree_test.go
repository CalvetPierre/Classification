package predictors_test

import (
	predictors "awesomeProject"
	"github.com/go-gota/gota/dataframe"
	"math/rand"
	"testing"
	"time"
)

func TestMakeTreeReg(t *testing.T) {
	//DT := NewDecisionTree(10)

	DT := new(predictors.DecisionTreeReg)
	DT.MaxDepth = 10

	err := DT.SetMinNodeSplitReg(0.20)
	if err != nil {
		return
	}

	// Import df from CSV
	xDF, yDF, err := ml.ImportTest()
	if err != nil {
		t.Error("Error importing the df: ", err)
	}

	err = DT.MakeTreeReg(xDF, yDF)
	if err != nil {
		t.Error("Error in make tree", err)
	}

	df := dataframe.LoadRecords(
		[][]string{
			{"X"},
			{"3"},
			{"6"},
			{"9"},
			{"26"},
		},
	)

	result := predictors.PredictReg(DT, &df)

	t.Log(result)
}

func TestMakeJungleReg(t *testing.T) {
	rand.Seed(time.Now().UTC().UnixNano())

	xDF, yDF, err := ml.ImportTest()
	if err != nil {
		t.Error("Error importing the df: ", err)
	}
	JG := new(predictors.JungleReg)

	err = JG.MakeJungleReg(xDF, yDF, 100, 20, 10, 0.05)
	if err != nil {
		t.Error("Error making jungle", err)
	}

	df := dataframe.LoadRecords(
		[][]string{
			{"X"},
			{"3"},
			{"6"},
			{"9"},
			{"26"},
		},
	)

	res := predictors.PredictJungleReg(JG, &df)

	t.Log(res)
}
