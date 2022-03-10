Environment1 is example environment 1 of circulant dynamics and 
Environment2 is example environment 2 of circulant dynamics with Restart

## Train
```
python train.py --env Environment_ID
```
Environment_ID is an integer equal to 1 for Environment1 and 2 for Environment2

`New_Whittle` is the approach where we model Whittle Indices as an Function Approximation using Neural Network.\
`Whittle` is another approach where Whittle Indices is modeled as an polynomial basis of the states. 

NOTE: `New_Whittle` is widely tested and all results in the report are using it.
