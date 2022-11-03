package predictors_test

import (
	"math/rand"
	"strings"
	"testing"
	"time"

	"github.com/go-gota/gota/dataframe"

)

func TestMakeTree(t *testing.T) {
	//DT := NewDecisionTree(10)

	DT := new(predictors.DecisionTree)
	DT.MaxDepth = 10

	err := DT.SetMinNodeSplit(0.05)
	if err != nil {
		return
	}

	// Import df from CSV
	xDF, yDF, err := ml.ImportIris()
	if err != nil {
		t.Error("Error importing the df: ", err)
	}

	err = DT.MakeTree(xDF, yDF)
	if err != nil {
		t.Error("Error in make tree", err)
	}

	df := dataframe.LoadRecords(
		[][]string{
			{"sepal.length", "sepal.width", "petal.length", "petal.width"},
			{"5.8", "4", "1.2", ".2"},
			{"6.1", "2.8", "4.7", "1.2"},
			{"6.3", "3.4", "5.6", "2.4"},
			{"5.1", "3.7", "1.5", ".4"},
		},
	)

	result := predictors.Predict(DT, &df)
	RealRes := []string{"Setosa", "Versicolor", "Virginica", "Setosa"}

	for i := 0; i < len(result); i++ {
		if strings.Compare(result[i], RealRes[i]) != 0 {
			t.Error("Error in predict")
			t.Log("expected", RealRes)
			t.Log("got : ", result)
			break
		}
	}
}

func TestMakeJungle(t *testing.T) {
	rand.Seed(time.Now().UTC().UnixNano())

	xDF, yDF, err := ml.ImportIris()
	if err != nil {
		t.Error("Error importing the df: ", err)
	}
	JG := new(predictors.Jungle)

	err = JG.MakeJungle(xDF, yDF, 5, 100, 10, 0.05)
	if err != nil {
		t.Error("Error making jungle", err)
	}

	df := dataframe.LoadRecords(
		[][]string{
			{"sepal.length", "sepal.width", "petal.length", "petal.width"},
			{"5.8", "4", "1.2", ".2"},
			{"6.1", "2.8", "4.7", "1.2"},
			{"6.3", "3.4", "5.6", "2.4"},
			{"5.1", "3.7", "1.5", ".4"},
		},
	)

	ExpectedRes := []string{"Setosa", "Versicolor", "Virginica", "Setosa"}
	res := predictors.PredictJungle(JG, &df)

	for i := 0; i < len(ExpectedRes); i++ {
		if strings.Compare(ExpectedRes[i], res[i]) != 0 {
			t.Error("Error in predict")
			t.Log("expected", ExpectedRes)
			t.Log("got : ", res)
			break
		}
	}
}

func TestPropClass(t *testing.T) {
	targetClass := "Léon"
	listIndex := []int{1, 3, 5, 7, 9, 10}
	df := dataframe.LoadRecords(
		[][]string{
			{"X"},
			{"PasLéon"},
			{"Léon"}, //1
			{"PasLéon"},
			{"Léon"}, //3
			{"Léon"},
			{"PasLéon"}, //5
			{"Léon"},
			{"Léon"}, //7
			{"Léon"},
			{"Léon"},    //9
			{"PasLéon"}, //10
		},
	)

	res, err := predictors.PropClass(targetClass, listIndex, &df)
	if err != nil {
		t.Error("Error running PropClass ", err)
	}

	if res != 4./6. {
		t.Error("Wrong predicted value")
		t.Log("Found          : ", res)
		t.Log("Expected around: ", 4./6.)
	}
}

func TestGini(t *testing.T) {
	AllClass := []string{"Léon", "PasLéon", "DemiLéon"}
	listIndex := []int{1, 3, 5, 7, 9, 10}
	df := dataframe.LoadRecords(
		[][]string{
			{"X"},
			{"Léon"},
			{"Léon"}, //1
			{"Léon"},
			{"DemiLéon"}, //3
			{"Léon"},
			{"Léon"}, //5
			{"Léon"},
			{"Léon"}, //7
			{"Léon"},
			{"Léon"},    //9
			{"PasLéon"}, //10
		},
	)

	res, err := predictors.Gini(AllClass, listIndex, &df)
	if err != nil {
		t.Error("Error running Gini ", err)
	}

	if res != 0.5 {
		t.Error("Wrong predicted value")
		t.Log("Found          : ", res)
		t.Log("Expected around: ", 0.5)
	}
}

func TestStatMoy(t *testing.T) {
	AllClass := []string{"Léon", "PasLéon", "DemiLéon"}
	listIndexDad := []int{1, 2, 4, 5, 6, 7, 9, 10}
	listIndexLeft := []int{2, 4, 6}
	listIndexRight := []int{1, 5, 7, 9, 10}
	df := dataframe.LoadRecords(
		[][]string{
			{"X"},
			{"DemiLéon"}, //0 , none
			{"DemiLéon"}, //1 , R
			{"Léon"},     //2 , L
			{"Léon"},     //3 , none
			{"Léon"},     //4 , L
			{"PasLéon"},  //5 , R
			{"Léon"},     //6 , L
			{"DemiLéon"}, //7 , R
			{"PasLéon"},  //8 , none
			{"DemiLéon"}, //9 , R
			{"Léon"},     //10 , R
		}, // L: 3 Léon
		// R: 1 Léon, 1 PasLéon, 3 DemiLéon
	)

	res, err := predictors.StatMoy(AllClass, listIndexDad, listIndexLeft, listIndexRight, &df)
	if err != nil {
		t.Error("Error running StatMoy ", err)
	}

	if res != 0.35 {
		t.Error("Wrong predicted value")
		t.Log("Found          : ", res)
		t.Log("Expected around: ", 0.35)
	}
}

func TestDeltaGini(t *testing.T) {
	AllClass := []string{"Léon", "PasLéon", "DemiLéon"}
	listIndexDad := []int{1, 2, 4, 5, 6, 7, 9, 10}
	listIndexLeft := []int{2, 4, 6}
	listIndexRight := []int{1, 5, 7, 9, 10}
	df := dataframe.LoadRecords(
		[][]string{
			{"X"},
			{"DemiLéon"}, //0 , none
			{"DemiLéon"}, //1 , R
			{"Léon"},     //2 , L
			{"Léon"},     //3 , none
			{"Léon"},     //4 , L
			{"PasLéon"},  //5 , R
			{"Léon"},     //6 , L
			{"DemiLéon"}, //7 , R
			{"PasLéon"},  //8 , none
			{"DemiLéon"}, //9 , R
			{"Léon"},     //10 , R
		}, // L: 3 Léon
		// R: 1 Léon, 1 PasLéon, 3 DemiLéon
	)

	res, err := predictors.DeltaGini(AllClass, listIndexDad, listIndexLeft, listIndexRight, &df)
	if err != nil {
		t.Error("Error running DeltaGini ", err)
	}

	if res != 0.24375000000000002 {
		t.Error("Wrong predicted value")
		t.Log("Found          : ", res)
		t.Log("Expected around: ", 0.24375000000000002)
	}
}

func TestMinmax(t *testing.T) {
	list := []float64{44.3, 33.7, 45.6, 66.4, 74.3, 84.7, 64.3, 83.8, 26.2, 85, 85, 25.1, 45, 78, 47}
	//25.1 in min
	//85 in max

	min, max := predictors.Minmax(list)

	if min != 25.1 || max != 85 {
		t.Error("Wrong predicted value")
		t.Log("Found          : ", min, max)
		t.Log("Expected : ", 25.1, 85)
	}
}

func TestTranspose(t *testing.T) {
	test := [][]string{
		{"a", "aa"},
		{"b", "ab"},
		{"c", "ac"},
		{"d", "ad"},
	}

	res := predictors.Transpose(test)
	ExpectedRes := [][]string{
		{"a", "b", "c", "d"},
		{"aa", "ab", "ac", "ad"},
	}

	for i := 0; i < len(res); i++ {
		for j := 0; j < len(res[0]); j++ {
			if strings.Compare(res[i][j], ExpectedRes[i][j]) != 0 {
				t.Error("Wrong predicted value")
				t.Log("Found          : ", res)
				t.Log("Expected : ", ExpectedRes)
				break
			}
		}
	}
}

func TestTargetMaj(t *testing.T) {
	test := dataframe.LoadRecords(
		[][]string{
			{"StringTest"},
			{"3"},
			{"2"},
			{"2"},
			{"1"},
			{"3"},
			{"a"}, // 5
			{"b"}, // 6
			{"c"}, // 7
			{"3"},
		},
	)

	node := new(predictors.TreeNode)
	node.ElementIndex = []int{0, 1, 2, 3, 4, 8}

	res, err := predictors.TargetMaj(node, &test)
	if err != nil {
		t.Error("error running TargetMaj")
	}

	if res != "3" {
		t.Error("Wrong predicted value")
		t.Log("Expected 3")
		t.Log("Got", res)
	}
}

func TestVote(t *testing.T) {
	test := [][]string{
		{"1", "2", "3", "4", "5"},
		{"2", "1", "3", "4", "5"},
		{"3", "2", "4", "5", "1"},
		{"1", "3", "2", "4", "5"},
		{"1", "2", "3", "5", "4"},
	}

	res := predictors.Vote(test)
	realRes := []string{"1", "2", "3", "4", "5"}
	for i := 0; i < len(res); i++ {
		if strings.Compare(res[i], realRes[i]) != 0 {
			t.Error("Wrong predicted value")
			t.Log("Expected ", realRes)
			t.Log("Got", res)
			break
		}
	}
}
