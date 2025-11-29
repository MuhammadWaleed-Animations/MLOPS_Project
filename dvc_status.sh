#!/bin/bash
# Quick DVC Commands for Building Classifier Project

echo "================================"
echo "DVC Status Check"
echo "================================"
dvc status

echo ""
echo "================================"
echo "DVC Tracked Files"
echo "================================"
echo "- data/ (Training images)"
echo "- artifacts/ (Models)"

echo ""
echo "================================"
echo "Common Commands"
echo "================================"
echo "Track new data:      dvc add data"
echo "Update tracking:     dvc add artifacts"
echo "Check status:        dvc status"
echo "Push to remote:      dvc push"
echo "Pull from remote:    dvc pull"

echo ""
echo "================================"
echo "After Training Workflow"
echo "================================"
echo "1. python train.py"
echo "2. dvc add artifacts"
echo "3. git add artifacts.dvc"
echo "4. git commit -m 'Updated model'"
echo "5. dvc push (if remote configured)"

echo ""
echo "Current DVC configuration:"
dvc remote list

