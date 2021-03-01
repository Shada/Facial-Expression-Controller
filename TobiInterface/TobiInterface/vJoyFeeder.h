#pragma once

// Monitor Force Feedback (FFB) vJoy device
#include "stdafx.h"

#include "public.h"
#include <malloc.h>
#include <string.h>
#include <stdlib.h>
#include "vjoyinterface.h"
#include <math.h>
#include <bitset>

class vJoyFeeder
{
public:
	vJoyFeeder();
	~vJoyFeeder();

	bool initialize(int _dev_ID = 1);
	void shutdown();

	void setWAxisZ(long z_value);
	void setWAxisY(long y_value);
	void setWAxisX(long x_value);

	void setBtn(int button_id, bool pressed);


	void sendMessage();
private:
	// Prototypes

	UINT DevID;

	BOOL PacketType2Str(FFBPType Type, LPTSTR Str);
	BOOL EffectType2Str(FFBEType Ctrl, LPTSTR Str);
	BOOL DevCtrl2Str(FFB_CTRL Type, LPTSTR Str);
	int  Polar2Deg(BYTE Polar);
	int  Byte2Percent(BYTE InByte);
	int TwosCompByte2Int(BYTE in);
	int TwosCompWord2Int(WORD in);
	int Deg2Pol(int deg);
	int serial_result = 0;



	JOYSTICK_POSITION_V2 iReport; // The structure that holds the full position data
};

