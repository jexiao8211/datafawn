# 1: Build soundscape into pipeline

instance variables:
- soundscape_config: solely defines the soundscape
    - event to soundscape map
    - backing track
    - other stuff tbd

soundscape_config = {
    event_sound_map = {
        event1: sound1,
        event2: sound2,
        ...
    },
    ...
}

runtime variables:
- events (defined by pipeline)
- input_vid_path (defined by pipeline)
- output_path (arg provided to pipeline.run) (call it soundscape_dest or something)

# 2: Build out soundscape more
- add backing track
    - add controls for how to handle it if its shorter/ longer than the original video
 -pitch up / down settings for each sound as part of config?


 - add validation for soundscape config in the soundscape gen constructors