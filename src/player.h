#ifndef PY4PD_PLAYER_H
#define PY4PD_PLAYER_H

#include "py4pd.h"

void PY4PD_Player_InsertThing(t_py *x, int onset, PyObject *value);
KeyValuePair* PY4PD_Player_GetValue(Dictionary* dictionary, int onset);

void py4pdPlay(t_py *x);
void py4pdStop(t_py *x);
void py4pdClear(t_py *x);

#endif
