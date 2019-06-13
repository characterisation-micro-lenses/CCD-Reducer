from ErrorGenerator import Error
import pytest

class ErrorA(Error):
    pass

def test_setup():
    with pytest.raises(ErrorA):
        raise ErrorA

def test_message():
    message = "blabla"
    with pytest.raises(ErrorA, match=message):
        raise ErrorA(message)

def test_inheritance():
    assert issubclass(ErrorA, Exception)
