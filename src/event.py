from collections import defaultdict

subcribers = defaultdict(list)

def subscribe(event_type: str, fn) -> None:
    if subcribers[event_type] and fn in subcribers[event_type]:
        print(f"Given function is already registered on {event_type} event")
        return
    subcribers[event_type].append(fn)
    
    
def unsubcribe(event_type: str, fn) -> None:
    if subcribers[event_type] and fn not in subcribers[event_type]:
        print(f"Given function is has not registered on {event_type} event")
        return
    subcribers[event_type].remove(fn)
    
    
def post_event(event_type: str, data=None) -> bool:
    if event_type not in subcribers:
        print(f"Event {event_type} is not registered")
        return False
    for fn in subcribers[event_type]:
        if data is None:
            fn()
        else:
            fn(data)
    return True