#!/usr/bin/env python3
"""Debug the enum decoding issue"""

import msgpack

# This is what we're seeing in the database:
raw_enum_data = b'\x93\xb5src.state.agent_state\xa4Move\xa6defect'

# Let's decode it step by step
print("🔍 DEBUGGING ENUM DECODING")
print("=" * 40)

try:
    # First, let's see what msgpack.unpackb gives us
    decoded = msgpack.unpackb(raw_enum_data, raw=False, strict_map_key=False)
    print(f"Raw decode result: {decoded}")
    print(f"Type: {type(decoded)}")
    
    if isinstance(decoded, list) and len(decoded) >= 3:
        print(f"Element 0 (class): {decoded[0]}")
        print(f"Element 1 (enum name): {decoded[1]}")  
        print(f"Element 2 (value): {decoded[2]}")
        print(f"✅ The value we want is: '{decoded[2]}'")
    
except Exception as e:
    print(f"❌ Decode failed: {e}")

# Now let's test our hook function
def test_decode_hook(code, data):
    print(f"\n🔧 TESTING DECODE HOOK")
    print(f"Code: {code}")
    print(f"Data type: {type(data)}")
    
    if code == 0:  # Enum code
        try:
            decoded = msgpack.unpackb(data, raw=False, strict_map_key=False)
            print(f"Hook decoded: {decoded}")
            
            if isinstance(decoded, list) and len(decoded) >= 3:
                return decoded[2]  # Return the enum value
            return decoded
        except Exception as e:
            print(f"Hook failed: {e}")
            return {"enum_decode_error": str(e)}
    
    return msgpack.ExtType(code, data)

# Test with ExtType
ext_type_data = msgpack.ExtType(0, raw_enum_data)
result = test_decode_hook(0, raw_enum_data)
print(f"\n✅ Final result should be: {result}")