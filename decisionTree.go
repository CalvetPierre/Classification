package predictors

import (
	"log"
	"math"
	"math/rand"
	"time"

	"github.com/go-gota/gota/dataframe"
)

// TODO Verification : When making a Jungle & choosing randomly target, does it impact the Gini Score and if Yes should it ?

// Jungle contains fields that can be used to make a jungle of tree.
type Jungle struct {
	Trees        []DecisionTree
	MaxDepth     int
	MinNodeSplit float64
}

// DecisionTree contains fields that can be used to make a tree.
// The last three fields are only used when making a Jungle
type DecisionTree struct {
	target       []string
	MaxDepth     int
	Nodes        []TreeNode
	MinNodeSplit float64
	InJungle     bool
	IndexForRoot []int
}

// TreeNode contains either two TreeNode (son) or a prediction (Leaf).
// The split is made on the variable TargetVar with a Threshold.
type TreeNode struct {
	Depth        int
	ElementIndex []int
	LeftNode     *TreeNode
	RightNode    *TreeNode
	LeafPred     string
	TargetVar    string
	Threshold    float64
	MinNodeSplit float64
	InJungle     bool
}

// NewDecisionTree initialize a new Decision Tree.
func NewDecisionTree(maxDepth int) DecisionTree { // nolint

	return DecisionTree{MaxDepth: maxDepth, MinNodeSplit: 0}
}

// NewTreeNode initialize a new node in a tree.
func NewTreeNode() TreeNode { // nolint

	return TreeNode{}
}

// SetMinNodeSplit allow you to modify the MinNodeSplit.
// If you want to stop a split at a node when there is only 5% of the remaining element set it to 0.05.
// It allow you to avoid overFitting your tree.
func (DT *DecisionTree) SetMinNodeSplit(t float64) error { // nolint
	if t <= 0 || t > 1 {
		return errors.ErrorValue
	}

	DT.MinNodeSplit = t

	return nil
}

// MakeJungle makes a jungle of tree.
func (Forest *Jungle) MakeJungle(xDF, yDF *dataframe.DataFrame, NbTree int, NbEch int, maxDepth int, MinNodeSplit float64) error {

	if NbEch > xDF.Nrow() {
		return errors.Error{String: "NbEch > xDF.NRow"}
	}

	for i := 0; i < NbTree; i++ {
		Forest.Trees = append(Forest.Trees, *new(DecisionTree))
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
		err = Forest.Trees[i].MakeTree(xDF, yDF)
		if err != nil {
			return err
		}
	}
	return nil
}

// MakeTree takes two df of attributes (xDF) & targets (yDF) and creates a Decision Tree.
func (DT *DecisionTree) MakeTree(xDF, yDF *dataframe.DataFrame) error { // nolint
	root := new(TreeNode)
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

	root, err := splitter(DT.target, root, DT.MaxDepth, xDF, yDF)
	if err != nil {
		return err
	}

	DT.Nodes = append(DT.Nodes, *root)

	return nil
}

// splitter split or do not split.
func splitter(AllTarget []string, node *TreeNode, maxDepth int, xDF, yDF *dataframe.DataFrame) (*TreeNode, error) {
	if node.Depth >= maxDepth || float64(len(node.ElementIndex)) < node.MinNodeSplit*float64(yDF.Nrow()) {
		var err error
		node.LeafPred, err = TargetMaj(node, yDF)
		if err != nil {
			return nil, err
		}
		//log.Println(node.ElementIndex)
		//log.Println(node.LeafPred)
		return node, nil
	}

	score, threshold, targetVar, err := optiTargetThreshold(AllTarget, node, xDF, yDF)
	if err != nil {
		return nil, err
	}

	if score < 0.05 {
		var err error
		node.LeafPred, err = TargetMaj(node, yDF)
		if err != nil {
			return nil, err
		}
		//log.Println(node.ElementIndex)
		//log.Println(node.LeafPred)
		return node, nil
	}

	nodeLeft := new(TreeNode)
	nodeRight := new(TreeNode)

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

	node.LeftNode, err = splitter(AllTarget, node.LeftNode, maxDepth, xDF, yDF)
	if err != nil {
		return nil, err
	}
	node.RightNode, err = splitter(AllTarget, node.RightNode, maxDepth, xDF, yDF)
	if err != nil {
		return nil, err
	}

	return node, nil
}

// Predict predicts the class of a given dataset through tree.
func Predict(tree *DecisionTree, xDFPred *dataframe.DataFrame) []string {
	var res []string
	for i := 0; i < xDFPred.Nrow(); i++ {
		name := WhatAmI(&tree.Nodes[0], *xDFPred, i)
		res = append(res, name)
	}
	//log.Println(res)
	return res
}

// PredictJungle predict the class of a given dataset using a random forest.
func PredictJungle(Jungle *Jungle, xDFPred *dataframe.DataFrame) []string {
	var res = make([][]string, len(Jungle.Trees))

	for i, tree := range Jungle.Trees {
		res[i] = Predict(&tree, xDFPred)
	}
	class := Vote(res)

	return class
}

// Vote takes a 2d array of string which represent the predicted class of multiple element on multiple decision tree.
// Vote returns an array which represent the predicted class of multiple element after a vote between the decision tree.
func Vote(TreePredictions [][]string) []string {
	VoteRes := make([]string, len(TreePredictions[0]))
	ListOrder := Transpose(TreePredictions)
	for i, pred := range ListOrder {
		var AllTarget []string
		var nbTarget []int
		var res string
		max := 0

		for _, elem := range pred {
			if isin(AllTarget, elem) == false {
				AllTarget = append(AllTarget, elem)
				nbTarget = append(nbTarget, 1)
			} else {
				if len(nbTarget) == 0 {
					log.Println("This leaf is empty! Error to fix!")
					return nil
				}

				for j, target := range AllTarget {
					if target == elem {
						nbTarget[j] = nbTarget[j] + 1
					}
				}
			}
		}

		for j := 0; j < len(nbTarget); j++ {
			if nbTarget[j] > max {
				max = nbTarget[j]
				res = AllTarget[j]
			}
		}

		VoteRes[i] = res
	}

	return VoteRes
}

// Transpose takes a 2d array of string and returns the transposed 2d array.
func Transpose(a [][]string) [][]string {
	n := len(a)    // row
	m := len(a[0]) // column

	b := make([][]string, m)

	for i := 0; i < m; i++ { // column with i
		b[i] = make([]string, n)
		for j := 0; j < n; j++ { // row with j
			b[i][j] = a[j][i]
		}
	}
	return b
}

// WhatAmI returns the target of the predicted element.
func WhatAmI(node *TreeNode, xDFPred dataframe.DataFrame, index int) string {
	if node.LeafPred != "" {
		//log.Println("you are a", node.LeafPred)
		return node.LeafPred
	}

	if xDFPred.Col(node.TargetVar).Elem(index).Float() < node.Threshold {
		return WhatAmI(node.LeftNode, xDFPred, index)
	} else {
		return WhatAmI(node.RightNode, xDFPred, index)
	}
}

// RdmAList shuffle a list of int if target = FALSE, string if target = TRUE.
func RdmAList(yDF *dataframe.DataFrame, NbEch int, target bool) ([]int, []string, error) {
	rand.Seed(time.Now().UTC().UnixNano())
	switch target {
	case false:
		if yDF.Nrow() < NbEch {
			return nil, nil, errors.ErrorValue
		}

		var all []int
		for i := 0; i < yDF.Nrow(); i++ {
			all = append(all, i)
		}
		rand.Shuffle(len(all), func(i, j int) { all[i], all[j] = all[j], all[i] })
		res := all[:NbEch]

		return res, nil, nil

	case true:
		allT := allTarget(yDF)

		if len(allT) < NbEch {
			return nil, nil, errors.ErrorValue
		}

		rand.Shuffle(len(allT), func(i, j int) { allT[i], allT[j] = allT[j], allT[i] })
		res := allT[:NbEch]

		return nil, res, nil
	}

	return nil, nil, nil
}

// optiTargetThreshold find the best Threshold & Target to split on at a given node.
// It returns the score, threshold, targetVar, error.
func optiTargetThreshold(AllTarget []string, node *TreeNode, xDF, yDF *dataframe.DataFrame) (float64, float64, string, error) {
	var score, threshold float64
	var targetVar string

	for i, targetVarTemp := range xDF.Names() {

		scoreTemp, thresholdTemp, err := optimiseThreshold(AllTarget, targetVarTemp, node, xDF, yDF)
		if err != nil {
			return 0, 0, "", err
		}
		if scoreTemp > score || i == 0 {
			score = scoreTemp
			threshold = thresholdTemp
			targetVar = targetVarTemp
		}
	}

	return score, threshold, targetVar, nil
}

// optimiseThreshold finds the best threshold to split a node on a given "targetVariable".
func optimiseThreshold(AllTarget []string, targetVar string, node *TreeNode, xDF, yDF *dataframe.DataFrame) (float64,
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

		for _, j := range node.ElementIndex {
			if xDF.Col(targetVar).Elem(j).Float() < i {
				listL = append(listL, j)
			} else {
				listR = append(listR, j)
			}
		}
		deltaG, err := DeltaGini(AllTarget, node.ElementIndex, listL, listR, yDF)
		if err != nil {
			return 0, 0, err
		}
		if i == minT || deltaG > maxDeltaG {
			maxDeltaG = deltaG
			listTh = []float64{i}
		} else if deltaG == maxDeltaG {
			listTh = append(listTh, i)
		}
	}

	if len(listTh) == 0 {
		return 0, 0, nil
	}
	threshold := listTh[int(math.Abs(float64(len(listTh))/2.0))]

	return maxDeltaG, threshold, nil
}

// TargetMaj returns a string which is the majority of target in the node.
func TargetMaj(node *TreeNode, yDF *dataframe.DataFrame) (string, error) {
	var UniAllTarget []string
	var nbTarget []int
	var res string
	max := 0

	for _, i := range node.ElementIndex {
		if isin(UniAllTarget, yDF.Elem(i, 0).String()) == false {
			UniAllTarget = append(UniAllTarget, yDF.Elem(i, 0).String())
			nbTarget = append(nbTarget, 1)
		} else {
			if len(nbTarget) == 0 {
				return "This leaf is empty! Error!", nil
			}
			for j, target := range UniAllTarget {
				if target == yDF.Elem(i, 0).String() {
					nbTarget[j] = nbTarget[j] + 1
				}
			}
		}
	}

	for i := 0; i < len(nbTarget); i++ {
		if nbTarget[i] > max {
			max = nbTarget[i]
			res = UniAllTarget[i]
		}
	}

	return res, nil
}

// PropClass returns the proportion of element in yDF with index in listIndex that are class.
func PropClass(target string, listIndex []int, yDF *dataframe.DataFrame) (float64, error) {
	lenIndex := float64(len(listIndex))
	if lenIndex == 0 {
		return 0, nil
	}
	nbClass := 0.0
	for _, i := range listIndex {
		if target == yDF.Elem(i, 0).String() {
			nbClass++
		}
	}

	return nbClass / lenIndex, nil
}

// Gini returns the Gini coefficient of element in yDF with index in listIndex.
func Gini(AllTarget []string, listIndex []int, yDF *dataframe.DataFrame) (float64, error) {
	res := 1.0
	for _, class := range AllTarget {
		temp, err := PropClass(class, listIndex, yDF)
		if err != nil {
			return 0, err
		}
		res -= math.Pow(temp, 2)
	}
	if res < 0 {
		return 0, errors.ErrorValue
	}

	return res, nil
}

// StatMoy returns the statistic mean.
func StatMoy(AllTarget []string, listIndexDad, listIndexLeft, listIndexRight []int, yDF *dataframe.DataFrame) (float64,
	error) {
	propL := float64(len(listIndexLeft)) / float64(len(listIndexDad))

	propR := float64(len(listIndexRight)) / float64(len(listIndexDad))

	giniL, err := Gini(AllTarget, listIndexLeft, yDF)
	if err != nil {
		return 0, err
	}

	giniR, err := Gini(AllTarget, listIndexRight, yDF)
	if err != nil {
		return 0, err
	}

	res := propL*giniL + propR*giniR

	return res, nil
}

// TODO, instead of recalculating giniDad in DeltaGini, the Gini Score could be stored in the node struct.
// Issue : access to the dad node => New parameters in func expected.

// DeltaGini returns the variation of "impurity" between the dad & the sons tree node.
func DeltaGini(AllTarget []string, listIndexDad, listIndexLeft, listIndexRight []int, yDF *dataframe.DataFrame) (float64,
	error) {
	giniDad, err := Gini(AllTarget, listIndexDad, yDF)
	if err != nil {
		return 0, err
	}

	statSon, err := StatMoy(AllTarget, listIndexDad, listIndexLeft, listIndexRight, yDF)
	if err != nil {
		return 0, err
	}

	res := giniDad - statSon

	return res, nil
}

// allTarget takes a dataframe (1 column only!) and returns a list of string of all the different target in the df. **
func allTarget(yDF *dataframe.DataFrame) []string {
	var UniTarget []string
	for i := 0; i < yDF.Nrow(); i++ {
		if target := yDF.Col(yDF.Names()[0]).Records()[i]; i == 0 || !isin(UniTarget, target) {
			UniTarget = append(UniTarget, target)
		}
	}
	return UniTarget
}

// Minmax returns the min & max of a list []float64. duh. **
func Minmax(list []float64) (float64, float64) {
	var min, max float64
	for i, val := range list {
		if i == 0 || val < min {
			min = val
		}
		if i == 0 || val > max {
			max = val
		}
	}
	return min, max
}

// isin returns true if b is in a, false otherwise. **
func isin(list []string, elem string) bool {
	for _, c := range list {
		if c == elem {
			return true
		}
	}

	return false
}
