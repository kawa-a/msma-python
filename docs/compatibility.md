# R and Python compatibility

## Version 0.1 scope

| R msma 4.0 capability | Python 0.1.0rc1 |
|---|---|
| Main `msma()` entry point | Implemented for S4PCA |
| Single numeric X matrix | Implemented |
| One-dimensional or one-column Z | Implemented |
| Soft and hard-exclusive structures | Implemented |
| `sumabs`, `any`, `count` history | Implemented |
| Fixed iterations and projection deflation | Implemented |
| Training/new-data reconstruction | Implemented |
| Overlap measures | Implemented |
| Repeated-split model selection | Initial implementation |
| Standard PCA, supervised PCA, PLS | Reserved, not implemented |
| Multiblock analysis | Reserved, not implemented |

Seven frozen R-reference scenarios reproduce directly at `rtol=0` and
`atol=1e-12`. Three unsupervised sparse scenarios are explicitly tracked as
platform-sensitive after their first strictly matching component.
