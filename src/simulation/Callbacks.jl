"""
    save_to_disk(save_every_dt)

    Return a callback function that dumps the state to disk at the given interval of time is
    seconds.

Keyword arguments
=================

- `skip_first`: Skip the first `skip_first` seconds before activating the callback.

- `call_at_end`: Call this function at the end of the integration.

"""
function save_to_disk(save_every_dt; skip_first = false, call_at_end = false)
    return call_every_dt(
        save_to_disk_func,
        save_every_dt;
        skip_first,
        call_at_end,
    )
end
