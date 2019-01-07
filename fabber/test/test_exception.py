from fabber import FabberException

def test_msg_only():
    e = FabberException("err1")
    assert isinstance(e, RuntimeError)
    assert str(e) == "err1"
    assert e.errcode is None
    assert e.log is None

def test_msg_errcode():
    e = FabberException("err1", -99)
    assert isinstance(e, RuntimeError)
    assert str(e) == "-99: err1"
    assert e.errcode == -99
    assert e.log is None
    
def test_msg_log():
    e = FabberException("err1", log="logtext")
    assert isinstance(e, RuntimeError)
    assert str(e) == "err1"
    assert e.errcode is None
    assert e.log == "logtext"

def test_msg_log_errcode():
    e = FabberException("err1", log="logtext", errcode=-44)
    assert isinstance(e, RuntimeError)
    assert str(e) == "-44: err1"
    assert e.errcode == -44
    assert e.log == "logtext"
    
