'''
 * (C) Copyright 02/2017 
 *
 * Anfatec Instruments AG 
 * Melanchthonstr. 28 
 * 08606 Oelsnitz/i.V.
 * Germany
 * http://www.anfatec.de/
 *
 * Feel free to use it.
 *
 
'''

#!/usr/bin/env python
# Send DDE Execute command to running program

# copyright recipe-577654-1
# changed by Falk mailbox@anfatec.de

import SXMRemote
import ctypes
import threading
import time
#import win32event
from win32 import win32event #manu 
from ctypes import POINTER, WINFUNCTYPE, c_char_p, c_void_p, c_int, c_ulong, c_char_p
from ctypes.wintypes import BOOL, DWORD, BYTE, INT, LPCWSTR, UINT, ULONG
from ctypes import POINTER, byref, c_ulong
from ctypes.wintypes import BOOL, HWND, MSG, UINT
from ctypes import byref, create_string_buffer
import configparser

# DECLARE_HANDLE(name) typedef void *name;
HCONV     = c_void_p  # = DECLARE_HANDLE(HCONV)
HDDEDATA  = c_void_p  # = DECLARE_HANDLE(HDDEDATA)
HSZ       = c_void_p  # = DECLARE_HANDLE(HSZ)
LPBYTE    = c_char_p  # POINTER(BYTE)
LPDWORD   = POINTER(DWORD)
LPSTR    = c_char_p
ULONG_PTR = c_ulong

# See windows/ddeml.h for declaration of struct CONVCONTEXT
PCONVCONTEXT = c_void_p

DMLERR_NO_ERROR = 0

# Predefined Clipboard Formats
CF_TEXT         =  1
CF_BITMAP       =  2
CF_METAFILEPICT =  3
CF_SYLK         =  4
CF_DIF          =  5
CF_TIFF         =  6
CF_OEMTEXT      =  7
CF_DIB          =  8
CF_PALETTE      =  9
CF_PENDATA      = 10
CF_RIFF         = 11
CF_WAVE         = 12
CF_UNICODETEXT  = 13
CF_ENHMETAFILE  = 14
CF_HDROP        = 15
CF_LOCALE       = 16
CF_DIBV5        = 17
CF_MAX          = 18

DDE_FACK          = 0x8000
DDE_FBUSY         = 0x4000
DDE_FDEFERUPD     = 0x4000
DDE_FACKREQ       = 0x8000
DDE_FRELEASE      = 0x2000
DDE_FREQUESTED    = 0x1000
DDE_FAPPSTATUS    = 0x00FF
DDE_FNOTPROCESSED = 0x0000

DDE_FACKRESERVED  = (~(DDE_FACK | DDE_FBUSY | DDE_FAPPSTATUS))
DDE_FADVRESERVED  = (~(DDE_FACKREQ | DDE_FDEFERUPD))
DDE_FDATRESERVED  = (~(DDE_FACKREQ | DDE_FRELEASE | DDE_FREQUESTED))
DDE_FPOKRESERVED  = (~(DDE_FRELEASE))

XTYPF_NOBLOCK        = 0x0002
XTYPF_NODATA         = 0x0004
XTYPF_ACKREQ         = 0x0008

XCLASS_MASK          = 0xFC00
XCLASS_BOOL          = 0x1000
XCLASS_DATA          = 0x2000
XCLASS_FLAGS         = 0x4000
XCLASS_NOTIFICATION  = 0x8000

XTYP_ERROR           = (0x0000 | XCLASS_NOTIFICATION | XTYPF_NOBLOCK)
XTYP_ADVDATA         = (0x0010 | XCLASS_FLAGS)
XTYP_ADVREQ          = (0x0020 | XCLASS_DATA | XTYPF_NOBLOCK)
XTYP_ADVSTART        = (0x0030 | XCLASS_BOOL)
XTYP_ADVSTOP         = (0x0040 | XCLASS_NOTIFICATION)
XTYP_EXECUTE         = (0x0050 | XCLASS_FLAGS)
XTYP_CONNECT         = (0x0060 | XCLASS_BOOL | XTYPF_NOBLOCK)
XTYP_CONNECT_CONFIRM = (0x0070 | XCLASS_NOTIFICATION | XTYPF_NOBLOCK)
XTYP_XACT_COMPLETE   = (0x0080 | XCLASS_NOTIFICATION )
XTYP_POKE            = (0x0090 | XCLASS_FLAGS)
XTYP_REGISTER        = (0x00A0 | XCLASS_NOTIFICATION | XTYPF_NOBLOCK )
XTYP_REQUEST         = (0x00B0 | XCLASS_DATA )
XTYP_DISCONNECT      = (0x00C0 | XCLASS_NOTIFICATION | XTYPF_NOBLOCK )
XTYP_UNREGISTER      = (0x00D0 | XCLASS_NOTIFICATION | XTYPF_NOBLOCK )
XTYP_WILDCONNECT     = (0x00E0 | XCLASS_DATA | XTYPF_NOBLOCK)
XTYP_MONITOR         = (0x00F0 | XCLASS_NOTIFICATION | XTYPF_NOBLOCK)

XTYP_MASK            = 0x00F0
XTYP_SHIFT           = 4

TIMEOUT_ASYNC        = 0xFFFFFFFF

def get_winfunc(libname, funcname, restype=None, argtypes=(), _libcache={}):
    """Retrieve a function from a library, and set the data types."""
    from ctypes import windll

    if libname not in _libcache:
        _libcache[libname] = windll.LoadLibrary(libname)
    func = getattr(_libcache[libname], funcname)
    func.argtypes = argtypes
    func.restype = restype

    return func


DDECALLBACK = WINFUNCTYPE(HDDEDATA, UINT, UINT, HCONV, HSZ, HSZ, HDDEDATA, 
                          ULONG_PTR, ULONG_PTR)

class DDE(object):
    """Object containing all the DDE functions"""
    AccessData         = get_winfunc("user32", "DdeAccessData",          LPBYTE,   (HDDEDATA, LPDWORD))
    ClientTransaction  = get_winfunc("user32", "DdeClientTransaction",   HDDEDATA, (LPBYTE, DWORD, HCONV, HSZ, UINT, UINT, DWORD, LPDWORD))
    Connect            = get_winfunc("user32", "DdeConnect",             HCONV,    (DWORD, HSZ, HSZ, PCONVCONTEXT))
    CreateStringHandle = get_winfunc("user32", "DdeCreateStringHandleW", HSZ,      (DWORD, LPCWSTR, UINT))
    Disconnect         = get_winfunc("user32", "DdeDisconnect",          BOOL,     (HCONV,))
    GetLastError       = get_winfunc("user32", "DdeGetLastError",        UINT,     (DWORD,))
    Initialize         = get_winfunc("user32", "DdeInitializeW",         UINT,     (LPDWORD, DDECALLBACK, DWORD, DWORD))
    FreeDataHandle     = get_winfunc("user32", "DdeFreeDataHandle",      BOOL,     (HDDEDATA,))
    FreeStringHandle   = get_winfunc("user32", "DdeFreeStringHandle",    BOOL,     (DWORD, HSZ))
    QueryString        = get_winfunc("user32", "DdeQueryStringA",        DWORD,    (DWORD, HSZ, LPSTR, DWORD, c_int))
    UnaccessData       = get_winfunc("user32", "DdeUnaccessData",        BOOL,     (HDDEDATA,))
    Uninitialize       = get_winfunc("user32", "DdeUninitialize",        BOOL,     (DWORD,))

class DDEError(RuntimeError):
    """Exception raise when a DDE errpr occures."""
    def __init__(self, msg, idInst=None):
        if idInst is None:
            RuntimeError.__init__(self, msg)
        else:
            RuntimeError.__init__(self, "%s (err=%s)" % (msg, hex(DDE.GetLastError(idInst))))

class DDEClient(object):
    """The DDEClient class.

    This class is used to create and manage a connection to a DDE (Dynamic Data Exchange) 
    service/topic. To handle callbacks, subclass DDEClient and override the callback method.
    """

    def __init__(self, service, topic):
        """
        Initialize a connection to a DDE service/topic.

        Args:
            service (str): The name of the DDE service to connect to.
            topic (str): The topic within the DDE service to connect to.
        """
        from ctypes import byref

        # Initialize instance variables
        self._idInst = DWORD(0)  # DDE instance identifier
        self._hConv = HCONV()    # Handle to the DDE conversation

        # Set up the callback function for DDE events
        self._callback = DDECALLBACK(self._callback)

        # Initialize the DDEML (Dynamic Data Exchange Management Library)
        res = DDE.Initialize(byref(self._idInst), self._callback, 0x00000010, 0)
        if res != DMLERR_NO_ERROR:
            raise DDEError(f"Unable to register with DDEML (err={hex(res)})")

        # Create string handles for the service and topic
        hszService = DDE.CreateStringHandle(self._idInst, service, 1200)
        hszTopic = DDE.CreateStringHandle(self._idInst, topic, 1200)

        # Establish a conversation with the DDE server
        self._hConv = DDE.Connect(self._idInst, hszService, hszTopic, PCONVCONTEXT())
        
        # Free the string handles after use
        DDE.FreeStringHandle(self._idInst, hszTopic)
        DDE.FreeStringHandle(self._idInst, hszService)

        # Raise an error if the conversation could not be established
        if not self._hConv:
            raise DDEError("Unable to establish a conversation with server", self._idInst)

        # Set up advisory links for specific topics
        self.advise("Scan")
        self.advise("Command")
        self.advise("SaveFileName")
        self.advise("ScanLine")
        self.advise("MicState")
        self.advise("SpectSave")

        # Initialize configuration parser
        self.config = configparser.ConfigParser()

        # Initialize variables for tracking responses
        self.NotGotAnswer = False  # Flag to indicate if an answer was received
        self.LastAnswer = ""       # Stores the last received answer

        
    def __del__(self):
        """Cleanup any active connections."""
        if self._hConv:
            DDE.Disconnect(self._hConv)  # Disconnect from the DDE conversation
        if self._idInst:
            DDE.Uninitialize(self._idInst)  # Uninitialize the DDE instance

    def advise(self, item, stop=False):
        """Request updates when DDE data changes."""

        hszItem = DDE.CreateStringHandle(self._idInst, item, 1200)  # Create a string handle for the item
        hDdeData = DDE.ClientTransaction(LPBYTE(), 0, self._hConv, hszItem, CF_TEXT, 
                                         XTYP_ADVSTOP if stop else XTYP_ADVSTART, TIMEOUT_ASYNC, LPDWORD())
        DDE.FreeStringHandle(self._idInst, hszItem)  # Free the string handle after use
        if not hDdeData:
            raise DDEError("Unable to %s advise" % ("stop" if stop else "start"), self._idInst)
        DDE.FreeDataHandle(hDdeData)  # Free the data handle

    def execute(self, command, timeout=5000):
        """Execute a DDE command."""
        self.NotGotAnswer = True  # Flag to indicate if an answer was received
        command = f'begin\r\n  {command}\r\nend.\r\n'  # Create a Pascal-style program
        command = bytes(command, 'utf-16').strip(b"\xff").strip(b"\xfe")  # Convert to bytes and clean up
        pData = c_char_p(command)
        cbData = DWORD(len(command) + 1)
        hDdeData = DDE.ClientTransaction(pData, cbData, self._hConv, HSZ(), CF_TEXT, XTYP_EXECUTE, timeout, LPDWORD())
        if not hDdeData:
            raise DDEError("Unable to send command", self._idInst)
        DDE.FreeDataHandle(hDdeData)  # Free the data handle

    def request(self, item, timeout=5000):
        """Request data from DDE service."""

        hszItem = DDE.CreateStringHandle(self._idInst, item, 1200)  # Create a string handle for the item
        hDdeData = DDE.ClientTransaction(LPBYTE(), 0, self._hConv, hszItem, CF_TEXT, XTYP_REQUEST, timeout, LPDWORD())
        DDE.FreeStringHandle(self._idInst, hszItem)  # Free the string handle after use
        if not hDdeData:
            raise DDEError("Unable to request item", self._idInst)

        if timeout != TIMEOUT_ASYNC:
            pdwSize = DWORD(0)
            pData = DDE.AccessData(hDdeData, byref(pdwSize))  # Access the data
            if not pData:
                DDE.FreeDataHandle(hDdeData)
                raise DDEError("Unable to access data", self._idInst)
            DDE.UnaccessData(hDdeData)  # Unaccess the data
        else:
            pData = None
        DDE.FreeDataHandle(hDdeData)  # Free the data handle
        return pData  # Return the requested data

    def callback(self, value, item=None):
        """Callback function for advice."""
        self.LastAnswer = value  # Store the last answer received
        if value.startswith(b'Scan on'):
            self.LastAnswer = 1
            self.ScanOnCallBack()  # Handle scan on callback
        elif value.startswith(b'Scan off'):
            self.LastAnswer = 0
            self.ScanOffCallBack()  # Handle scan off callback
        elif item.startswith(b'SaveFileName'):
            FileName = str(value, 'utf-8').strip('\r\n')
            self.SaveIsDone(FileName)  # Handle save completion
        elif item.startswith(b'ScanLine'):
            value = str(value, 'utf-8').strip('\r\n')
            self.Scan(value)  # Handle scan line
        elif item.startswith(b'MicState'):
            self.MicState(value)  # Handle microphone state
        elif item.startswith(b'SpectSave'):
            self.SpectSave(value)  # Handle spectrum save
        elif item.startswith(b'Command'):
            self.LastAnswer = value  # Echo of command
        else:
            print("Unknown callback %s: %s" % (item, value))  # Handle unknown callback

    def _callback(self, wType, uFmt, hConv, hsz1, hsz2, hDdeData, dwData1, dwData2):
        """Handle DDE callback events."""
        if wType == XTYP_XACT_COMPLETE:
            pass  # Transaction complete
        elif wType == XTYP_DISCONNECT:
            print('disconnect')  # Handle disconnection
        elif wType == XTYP_ADVDATA:

            dwSize = DWORD(0)
            pData = DDE.AccessData(hDdeData, byref(dwSize))  # Access the data
            if pData:
                item = create_string_buffer(128)  # Create a buffer for the item
                DDE.QueryString(self._idInst, hsz2, item, 128, 1004)  # Query the item string
                self.callback(pData, item.value)  # Call the callback function
                self.NotGotAnswer = False  # Reset the answer flag
                DDE.UnaccessData(hDdeData)  # Unaccess the data
            return DDE_FACK  # Acknowledge the data
        else:
            print('callback' + hex(wType))  # Handle other callback types

        return 0  # Default return value

    def ScanOnCallBack(self):
        """Handle scan on callback."""
        print('scan is on')

    def ScanOffCallBack(self):
        """Handle scan off callback."""
        print('scan is off')

    def SaveIsDone(self, FileName):
        """Handle completion of save operation."""
        print(FileName)

    def Scan(self, LineNr):
        """Handle scanning operation."""
        pass  # Placeholder for scan logic

    def MicState(self, Value):
        """Handle microphone state updates."""
        print('MicState ' + str(Value))

    def SpectSave(self, Value):
        """Handle spectrum save updates."""
        print('SpectSave ' + str(Value))

    def StartMsgLoop(self):
        """Start the message loop in a separate thread."""
        MsgLoop = MyMsgClass()
        MsgLoop.start()

    def GetIniEntry(self, section, item):
        """Retrieve a value from the INI configuration file."""
        IniName = self.request('IniFileName')  # Get the current INI file name
        IniName = str(IniName, 'utf-8').strip('\r\n')  # Convert to string and clean up
        self.config.read(IniName)  # Read the INI file
        val = self.config.get(section, item)  # Get the value from the specified section and item
        return val

    def GetChannel(self, ch):
        """Get the value of a specific channel."""
        string = f"a:=GetChannel({ch});\r\n  writeln(a);"  # Create the command string
        self.execute(string, 1000)  # Execute the command

        while self.NotGotAnswer:
            loop()  # Wait for the answer

        BackStr = self.LastAnswer  # Get the last answer
        BackStr = str(BackStr, 'utf-8').split('\r\n')  # Convert to string and split
        if len(BackStr) >= 2:
            NrStr = BackStr[1].replace(',', '.')  # Replace comma with dot
            val = float(NrStr)  # Convert to float
            return val  # Return the value
        return None  # Return None if no valid value

    def SendWait(self, command):
        """Send a command and wait for a response."""
        self.execute(command, 1000)  # Execute the command
        while self.NotGotAnswer:
            loop()  # Wait for the answer

    def GetPara(self, TopicItem):
        """Get a parameter value from the DDE service."""
        self.execute(TopicItem, 1000)  # Execute the command

        while self.NotGotAnswer:
            loop()  # Wait for the answer

        BackStr = self.LastAnswer  # Get the last answer
        BackStr = str(BackStr, 'utf-8').split('\r\n')  # Convert to string and split
        if len(BackStr) >= 2:
            NrStr = BackStr[1].replace(',', '.')  # Replace comma with dot
            val = float(NrStr)  # Convert to float
            return val  # Return the value
        return None  # Return None if no valid value

    def GetScanPara(self, item):
        """Get scan parameter value."""
        TopicItem = f"a:=GetScanPara('{item}');\r\n  writeln(a);"
        return self.GetPara(TopicItem)  # Retrieve the parameter value

    def GetFeedbackPara(self, item):
        """Get feedback parameter value."""
        TopicItem = f"a:=GetFeedPara('{item}');\r\n  writeln(a);"
        return self.GetPara(TopicItem)  # Retrieve the parameter value


class MyMsgClass(threading.Thread):
    """
    MyMsgClass is a subclass of threading.Thread that runs a Windows message loop
    in a separate thread. This class is useful for handling Windows messages 
    in a GUI application without blocking the main thread.

    Attributes:
        None

    Methods:
        run(): Starts the message loop that processes Windows messages.
    """
    
    def __init__(self):
        """
        Initializes the MyMsgClass instance and starts the threading.Thread.
        """
        threading.Thread.__init__(self)

    def run(self):
        """Run the main windows message loop."""
        # Import necessary ctypes components for Windows API calls


        # Define types for the Windows message structure and function return types
        LPMSG = POINTER(MSG)
        LRESULT = c_ulong
        
        # Get the function pointers for Windows API functions
        GetMessage = get_winfunc("user32", "GetMessageW", BOOL, (LPMSG, HWND, UINT, UINT))
        TranslateMessage = get_winfunc("user32", "TranslateMessage", BOOL, (LPMSG,))
        DispatchMessage = get_winfunc("user32", "DispatchMessageW", LRESULT, (LPMSG,))

        # Create a MSG structure to hold the message
        msg = MSG()
        lpmsg = byref(msg)
        
        print("Debug: Start Msg loop")
        
        # Start the message loop
        while GetMessage(lpmsg, HWND(), 0, 0) > 0:
            TranslateMessage(lpmsg)  # Translate the message to a more understandable format
            DispatchMessage(lpmsg)   # Dispatch the message to the appropriate window procedure
            print("loop")  # Debug output to indicate the loop is running


def loop():
    LPMSG = POINTER(MSG)
    LRESULT = c_ulong
    GetMessage = get_winfunc("user32", "GetMessageW", BOOL, (LPMSG, HWND, UINT, UINT))
    TranslateMessage = get_winfunc("user32", "TranslateMessage", BOOL, (LPMSG,))
    DispatchMessage = get_winfunc("user32", "DispatchMessageW", LRESULT, (LPMSG,))

    msg = MSG()
    lpmsg = byref(msg)
    GetMessage(lpmsg, HWND(), 0, 0)
    TranslateMessage(lpmsg)
    DispatchMessage(lpmsg)

MySXM= SXMRemote.DDEClient("SXM","Remote")

