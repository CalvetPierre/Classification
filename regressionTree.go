package predictors

import (
	"github.com/go-gota/gota/dataframe"
	"gonum.org/v1/gonum/floats"
	"math"
)

// JungleReg contains fields that can be used to make a jungle of tree.
type JungleReg struct {
	Trees        []DecisionTreeReg
	MaxDepth     int
	MinNodeSplit float64
}

// DecisionTreeReg contains fields that can be used to make a tree.
// The last three fields are only used when making a Jungle
type DecisionTreeReg struct {
	target       []string
	MaxDepth     int
	Nodes        []TreeNodeReg
	MinNodeSplit float64
	InJungle     bool
	IndexForRoot []int
}

// TreeNodeReg contains either two TreeNodeReg (son) or a prediction (Leaf).
// The split is made on the variable TargetVar with a Threshold.
// In TreeNodeReg, the LeafPred is a float64 and not a string.
type TreeNodeReg struct {
	Depth        int
	ElementIndex []int
	LeftNode     *TreeNodeReg
	RightNode    *TreeNodeReg
	LeafPred     float64
	TargetVar    string
	Threshold    float64
	MinNodeSplit float64
	InJungle     bool
}

// SetMinNodeSplitReg allow you to modify the MinNodeSplit.
// If you want to stop a split at a node when there is only 5% of the remaining element set it to 0.05.
// It allows you to avoid overFitting your tree.
func (DT *DecisionTreeReg) SetMinNodeSplitReg(t float64) error { // nolint
	if t <= 0 || t > 1 {
		return errors.ErrorValue
	}

	DT.MinNodeSplit = t

	return nil
}

// MakeJungleReg makes a jungle of tree.
func (Forest *JungleReg) MakeJungleReg(xDF, yDF *dataframe.DataFrame, NbTree int, NbEch int, maxDepth int, MinNodeSplit float64) error {

	if NbEch > xDF.Nrow() {
		return errors.Error{String: "NbEch > xDF.NRow"}
	}

	for i := 0; i < NbTree; i++ {
		Forest.Trees = append(Forest.Trees, *new(DecisionTreeReg))
		Forest.Trees[i].InJungle = true
		Forest.Trees[i].MaxDepth = maxDepth
		Forest.Trees[i].MinNodeSplit = MinNodeSplit
		index, _, err := RdmAList(yDF, NbEch, false)
		if err != nil {
			return err
		}
		Forest.Trees[i].IndexForRoot = index

		//log.Println("")
		//log.Println("New Tree in Jungle")
		err = Forest.Trees[i].MakeTreeReg(xDF, yDF)
		if err != nil {
			return err
		}
	}
	return nil
}

// MakeTreeReg takes two df of attributes (xDF) & targets (yDF) and creates a Decision Tree.
func (DT *DecisionTreeReg) MakeTreeReg(xDF, yDF *dataframe.DataFrame) error { // nolint
	root := new(TreeNodeReg)
	if DT.InJungle {
		NbTarget := int(math.Sqrt(float64(len(allTarget(yDF)))))
		_, target, err := RdmAList(yDF, NbTarget, true)
		if err != nil {
			return err
		}

		DT.target = target
		root.Depth = 0
		root.MinNodeSplit = DT.MinNodeSplit
		root.ElementIndex = DT.IndexForRoot
		root.InJungle = true
	} else {
		DT.target = allTarget(yDF)
		root.Depth = 0
		root.MinNodeSplit = DT.MinNodeSplit
		for i := 0; i < xDF.Nrow(); i++ {
			root.ElementIndex = append(root.ElementIndex, i)
		}
	}

	root, err := splitterReg(DT.target, root, DT.MaxDepth, xDF, yDF)
	if err != nil {
		return err
	}

	DT.Nodes = append(DT.Nodes, *root)

	return nil
}

// splitterReg split or do not split.
func splitterReg(AllTarget []string, node *TreeNodeReg, maxDepth int, xDF, yDF *dataframe.DataFrame) (*TreeNodeReg, error) {
	if node.Depth >= maxDepth || float64(len(node.ElementIndex)) < node.MinNodeSplit*float64(yDF.Nrow()) {
		var err error
		node.LeafPred, err = Average(node.ElementIndex, yDF)
		if err != nil {
			return nil, err
		}
		//log.Println(node.ElementIndex)
		//log.Println(node.LeafPred)
		return node, nil
	}

	_, threshold, targetVar, err := optiTargetThresholdReg(AllTarget, node, xDF, yDF)
	//log.Println(threshold,targetVar)
	if err != nil {
		return nil, err
	}

	nodeLeft := new(TreeNodeReg)
	nodeRight := new(TreeNodeReg)

	nodeRight.MinNodeSplit = node.MinNodeSplit
	nodeLeft.MinNodeSplit = node.MinNodeSplit

	nodeLeft.Depth = node.Depth + 1
	nodeRight.Depth = node.Depth + 1

	//log.Println(score, targetVar, threshold)

	for _, i := range node.ElementIndex {
		if xDF.Col(targetVar).Elem(i).Float() < threshold {
			nodeLeft.ElementIndex = append(nodeLeft.ElementIndex, i)
		} else {
			nodeRight.ElementIndex = append(nodeRight.ElementIndex, i)
		}
	}

	if node.InJungle {
		nodeRight.InJungle = true
		nodeLeft.InJungle = true
		NbTarget := int(math.Sqrt(float64(len(allTarget(yDF)))))
		_, AllTarget, err = RdmAList(yDF, NbTarget, true)
		if err != nil {
			return nil, err
		}
	}

	node.RightNode = nodeRight
	node.LeftNode = nodeLeft
	node.TargetVar = targetVar
	node.Threshold = threshold

	node.LeftNode, err = splitterReg(AllTarget, node.LeftNode, maxDepth, xDF, yDF)
	if err != nil {
		return nil, err
	}
	node.RightNode, err = splitterReg(AllTarget, node.RightNode, maxDepth, xDF, yDF)
	if err != nil {
		return nil, err
	}

	return node, nil
}

//optiTargetThresholdReg find the best Threshold & Target to split on at a given node.
// It returns the score, threshold, targetVar, error.
func optiTargetThresholdReg(AllTarget []string, node *TreeNodeReg, xDF, yDF *dataframe.DataFrame) (float64, float64, string, error) {
	var score, threshold float64
	var targetVar string

	for i, targetVarTemp := range xDF.Names() {

		scoreTemp, thresholdTemp, err := optimiseThresholdReg(targetVarTemp, node, xDF, yDF)
		if err != nil {
			return 0, 0, "", err
		}
		if scoreTemp < score || i == 0 {
			score = scoreTemp
			threshold = thresholdTemp
			targetVar = targetVarTemp
		}
	}

	return score, threshold, targetVar, nil
}

// optimiseThresholdReg finds the best threshold to split a node on a given "targetVariable".
func optimiseThresholdReg(targetVar string, node *TreeNodeReg, xDF, yDF *dataframe.DataFrame) (float64,
	float64, error) {
	var nodeDfElem []float64
	for _, i := range node.ElementIndex {
		nodeDfElem = append(nodeDfElem, xDF.Col(targetVar).Elem(i).Float())
	}
	minT, maxT := Minmax(nodeDfElem)
	nbT := 50.0
	stepT := (maxT - minT) / nbT

	var maxDeltaG float64
	var listTh []float64

	for i := minT; i < maxT; i = i + stepT {
		var listL, listR []int

		//log.Println(len(node.ElementIndex))
		for _, j := range node.ElementIndex {
			if xDF.Col(targetVar).Elem(j).Float() < i || xDF.Col(targetVar).Elem(j).Float() == minT {
				listL = append(listL, j)
			} else {
				listR = append(listR, j)
			}
		}
		//log.Println(len(listL))
		//log.Println(len(listR))
		RScore, err := RegScore(listL, listR, yDF)
		if err != nil {
			return 0, 0, err
		}
		if i == minT || RScore < maxDeltaG {
			maxDeltaG = RScore
			listTh = []float64{i}
		} else if RScore == maxDeltaG {
			listTh = append(listTh, i)
		}
	}

	if len(listTh) == 0 {
		return 0, 0, nil
	}
	threshold := listTh[int(math.Abs(float64(len(listTh))/2.0))]

	return maxDeltaG, threshold, nil
}

// Average returns the average of element in df which index are in node.
func Average(nodeIndex []int, yDF *dataframe.DataFrame) (float64, error) {
	var res float64
	var list []float64

	for _, index := range nodeIndex {
		list = append(list, yDF.Elem(index, 0).Float())
	}

	if len(list) == 0 {
		return 0, errors.Error{String: "Error running Average : nodeIndex is empty"}
	}

	res = floats.Sum(list)
	res = res / float64(len(list))

	return res, nil
}

func RegScore(listL, listR []int, yDF *dataframe.DataFrame) (float64, error) {
	var res float64
	AvgL, err := Average(listL, yDF)
	AvgR, err := Average(listR, yDF)

	for _, i := range listL {
		res += math.Pow(AvgL-yDF.Elem(i, 0).Float(), 2)
	}
	for _, j := range listR {
		res += math.Pow(AvgR-yDF.Elem(j, 0).Float(), 2)
	}

	return res, err
}

// PredictReg predicts the class of a given dataset through tree.
func PredictReg(tree *DecisionTreeReg, xDFPred *dataframe.DataFrame) []float64 {
	var res []float64
	for i := 0; i < xDFPred.Nrow(); i++ {
		val := WhatAmIReg(&tree.Nodes[0], *xDFPred, i)
		res = append(res, val)
	}
	return res
}

func WhatAmIReg(node *TreeNodeReg, xDFPred dataframe.DataFrame, index int) float64 {
	//log.Println(node.LeafPred)
	//log.Println(node.Threshold)
	if node.Threshold == 0 {
		return node.LeafPred
	}

	if xDFPred.Col(node.TargetVar).Elem(index).Float() < node.Threshold {
		return WhatAmIReg(node.LeftNode, xDFPred, index)
	} else {
		return WhatAmIReg(node.RightNode, xDFPred, index)
	}
}

// PredictJungleReg predict the class of a given dataset using a random forest.
func PredictJungleReg(Jungle *JungleReg, xDFPred *dataframe.DataFrame) []float64 {
	var res = make([][]float64, len(Jungle.Trees))

	for i, tree := range Jungle.Trees {
		res[i] = PredictReg(&tree, xDFPred)
	}
	class := VoteReg(res)

	return class
}

// VoteReg takes a 2d array of string which represent the predicted class of multiple element on multiple decision tree.
// VoteReg returns an array which represent the predicted class of multiple element after a vote between the decision tree.
func VoteReg(TreePredictions [][]float64) []float64 {
	VoteRes := make([]float64, len(TreePredictions[0]))
	ListOrder := Transpose64(TreePredictions)
	for i := 0; i < len(ListOrder); i++ {

		var res float64

		res = floats.Sum(ListOrder[i]) / (float64(len(ListOrder[i])))

		VoteRes[i] = res
	}

	return VoteRes
}

// Transpose64 takes a 2d array of string and returns the transposed 2d array.
func Transpose64(a [][]float64) [][]float64 {
	n := len(a)    // row
	m := len(a[0]) // column

	b := make([][]float64, m)

	for i := 0; i < m; i++ { // column with i
		b[i] = make([]float64, n)
		for j := 0; j < n; j++ { // row with j
			b[i][j] = a[j][i]
		}
	}
	return b
}
