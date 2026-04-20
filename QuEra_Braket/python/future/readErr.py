# pieces of code

i    parser.add_argument('--readErrEps', default=None, type=float, help='probability of state 1 to be measured as state 0')


    if args.readErrEps!=None:
        from ReadErrMit_MIS import  ReadErrorMitigation4MIS
        ana=ReadErrorMitigation4MIS(task)
        XXX add loop over solutions with 2.5 sigma certainty
        move hits to undo readerr
        compute effective eps per solution
        report unaccounted for hits
        #ana.find_similar('1000011')# mid=67 use MSBL convention
        #ana.find_similar('0100011')# mid=35
        ana.find_similar('0010010')# mid=18
        ana.find_measurements()
        exit(0)
