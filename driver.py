#!/usr/bin/env python
from __future__ import print_function
from structure_denoise.interface import parser, plot_wavelet_mode, plot_projection_mode, \
     diagnostic_mode, processing_mode

def main(parsed_args):
    skipped = []
    mode_to_callable = \
      {'process_mode': processing_mode,
       'plot_wavelet_mode': plot_wavelet_mode,
       'plot_projection_mode': plot_projection_mode,
       'diagnostic_mode': diagnostic_mode}

    for mode in mode_to_callable:
        if getattr(parsed_args, mode):
            try:
                mode_to_callable[mode](parsed_args)
            except Exception as e:
                if parsed_args.reraise:
                    raise e
                skipped.append(mode + ': ' + str(e))
            
    ## if parsed_args.plot_wavelet_mode:
    ##     try:
    ##         plot_wavelets_mode(parsed_args)
    ##     except Exception as e:
    ##         skipped.append('plot wavelet: ' + str(e))

    ## if parsed_args.plot_projection_mode:
    ##     try:
    ##         plot_projection_mode(parsed_args)
    ##     except Exception as e:
    ##         skipped.append('plot projection: ' + str(e))

    if skipped:
        print('\n'.join(['Exceptions:'] + skipped))


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
