#include "py4pd.h"
#include "utils.h"

// ======================================================
Dictionary *Py4pdPlayer_CreateDictionary() {
    Dictionary *dictionary = (Dictionary *)malloc(sizeof(Dictionary));
    dictionary->Entries = NULL;
    dictionary->Size = 0;
    dictionary->LastOnset = 0;
    return dictionary;
}

// ======================================================
int Py4pdPlayer_CompareOnset(const void *a, const void *b) {
    const KeyValuePair *entryA = (const KeyValuePair *)a;
    const KeyValuePair *entryB = (const KeyValuePair *)b;
    if (entryA->Onset < entryB->Onset)
        return -1;
    else if (entryA->Onset > entryB->Onset)
        return 1;
    else
        return 0;
}

// ======================================================
KeyValuePair *Py4pdPlayer_PlayerGetValue(Dictionary *dictionary, int onset) {
    if (dictionary == NULL) {
        return NULL;
    }
    for (int i = 0; i < dictionary->Size; i++) {
        if (dictionary->Entries[i].Onset == onset) {
            return &(dictionary->Entries[i]);
        }
    }
    return NULL; // Return NULL if no matching onset is found
}

// ======================================================
void Py4pdPlayer_PlayerInsertThing(t_py *x, int onset, PyObject *value) {
    if (x->PlayerDict == NULL) {
        x->PlayerDict = Py4pdPlayer_CreateDictionary();
    }
    int index = -1;
    for (int i = 0; i < x->PlayerDict->Size; i++) {
        if (x->PlayerDict->Entries[i].Onset == onset) {
            index = i;
            break;
        }
    }

    if (index != -1) {
        // Onset already exists in the dictionary, add value to the array
        x->PlayerDict->Entries[index].pValues =
            (PyObject **)realloc(x->PlayerDict->Entries[index].pValues,
                                 (x->PlayerDict->Entries[index].Size + 1) * sizeof(PyObject *));
        x->PlayerDict->Entries[index].pValues[x->PlayerDict->Entries[index].Size] = value;
        x->PlayerDict->Entries[index].Size++;
    } else {
        // Onset doesn't exist, create a new entry
        if (onset > x->PlayerDict->LastOnset) {
            x->PlayerDict->LastOnset = onset;
        }
        x->PlayerDict->Entries = (KeyValuePair *)realloc(
            x->PlayerDict->Entries, (x->PlayerDict->Size + 1) * sizeof(KeyValuePair));
        KeyValuePair *newEntry = &(x->PlayerDict->Entries[x->PlayerDict->Size]);
        newEntry->Onset = onset;
        newEntry->pValues = (PyObject **)malloc(sizeof(PyObject *));
        newEntry->pValues[0] = value;
        newEntry->Size = 1;
        x->PlayerDict->Size++;
    }
}

// ======================================================
void Py4pdPlayer_FreeDictionary(Dictionary *dictionary) {
    if (dictionary == NULL) {
        return;
    }
    for (int i = 0; i < dictionary->Size; i++) {
        free(dictionary->Entries[i].pValues);
    }
    free(dictionary->Entries);
    free(dictionary);
}

// ================== PLAYER ======================
void Py4pdPlayer_PlayTick(t_py *x) {
    if (x->PlayerDict->LastOnset > x->MsOnset) {
        clock_delay(x->PlayerClock, 1);
    } else {
        clock_unset(x->PlayerClock);
    }
    x->MsOnset++;
    KeyValuePair *entry = Py4pdPlayer_PlayerGetValue(x->PlayerDict, x->MsOnset);
    if (entry != NULL) {
        for (int i = 0; i < entry->Size; i++) {
            PyObject *pValue = Py_BuildValue("O", entry->pValues[i]);
            t_py4pd_pValue *pdPyValue = (t_py4pd_pValue *)malloc(sizeof(t_py4pd_pValue));
            pdPyValue->pValue = pValue;
            pdPyValue->ObjOwner = x->ObjName;
            Py4pdUtils_ConvertToPd(x, pdPyValue, x->MainOut);
            free(pdPyValue);
        }
    }
}

// ======================================================
void Py4pdPlayer_Play(t_py *x, t_symbol *s, int ac, t_atom *av) {
    (void)s;
    x->MsOnset = 0;
    if (x->PlayerDict == NULL) {
        pd_error(x, "[%s]: Nothing to play.", x->ObjName->s_name);
        return;
    }
    if (ac != 0 && av[0].a_type == A_FLOAT) {
        x->MsOnset = (int)av->a_w.w_float;
    }
    x->PlayerClock = clock_new(x, (t_method)Py4pdPlayer_PlayTick);
    Py4pdPlayer_PlayTick(x);
    return;
}

// ======================================================
void Py4pdPlayer_Stop(t_py *x) {
    if (x->PlayerClock == NULL) {
        pd_error(x, "[%s]: Nothing to stop.", x->ObjName->s_name);
        return;
    }
    clock_unset(x->PlayerClock);
    return;
}

// ======================================================
void Py4pdPlayer_Clear(t_py *x) {
    if (x->PlayerClock != NULL) {
        clock_unset(x->PlayerClock);
    }
    Py4pdPlayer_FreeDictionary(x->PlayerDict);
    x->PlayerDict = NULL;
    return;
}
