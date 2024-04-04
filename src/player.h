// clang-format off
#ifndef PY4PD_PLAYER_H
#define PY4PD_PLAYER_H


#include "py4pd.h"

void Py4pdPlayer_PlayerSendNewThing(t_py *x, int onset, t_symbol *receiver, PyObject *value); // TODO: Need documentation
void Py4pdPlayer_PlayerInsertThing(t_py *x, int onset, PyObject *value);
KeyValuePair* Py4pdPlayer_PlayerGetValue(Dictionary* dictionary, int onset);
void Py4pdPlayer_Play(t_py *x, t_symbol *s, int argc, t_atom *argv);
void Py4pdPlayer_Stop(t_py *x);
void Py4pdPlayer_Clear(t_py *x);
void *Py4pdLib_NewObj(t_symbol *s, int argc, t_atom *argv);

#endif




