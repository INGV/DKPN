# ON GAIA (SOME-ICE)


Main problems are:
1. windows seems to have the order of components missing
2. Seems the some how pre-process the stream by removeing AT LEAST the mean value
3. Check the custom `predict` class-method sliding samples

### COMPONENT ORDER
The defaults  of seisbench fro the `WaveformModel` are:
- `component_order`: 
- `flexible_horizontal_components`


Therefore these applies also to PhaseNet from pretrained INSTANCE (as they are not specified in the json file, and therefore doesn't overwrite the defauls)

```
# In model.base.py, the default for flexible.. is true. So it creates this match dicto --> final order ZNE
ipdb>  comp_dict

{'Z': 0, 'N': 1, 'E': 2, '1': 1}
```


No matter the order you provide (in this case ZNE), seisbench will flip it anyway to ENZ, so we should do the same!!!!
The `flexible_horizontal_components` parameter seems to not affect anything in this sense..so I'll remove it from the equation




when using DKPN make sure the CFs order is ok at the end of `stream_to_arrays` (annotate._async)
