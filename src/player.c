#include "py4pd.h"
#include "utils.h"

// ======================================================
Dictionary *Py4pdPlayer_CreateDictionary() {
    Dictionary *dictionary = (Dictionary *)malloc(sizeof(Dictionary));
    dictionary->entries = NULL;
    dictionary->size = 0;
    dictionary->lastOnset = 0;
    return dictionary;
}

// ======================================================
int Py4pdPlayer_CompareOnset(const void *a, const void *b) {
    const KeyValuePair *entryA = (const KeyValuePair *)a;
    const KeyValuePair *entryB = (const KeyValuePair *)b;
    if (entryA->onset < entryB->onset)
        return -1;
    else if (entryA->onset > entryB->onset)
        return 1;
    else
        return 0;
}

// ======================================================
KeyValuePair *Py4pdPlayer_PlayerGetValue(Dictionary *dictionary, int onset) {
    if (dictionary == NULL) {
        return NULL;
    }
    for (int i = 0; i < dictionary->size; i++) {
        if (dictionary->entries[i].onset == onset) {
            return &(dictionary->entries[i]);
        }
    }
    return NULL; // Return NULL if no matching onset is found
}

// ======================================================
void Py4pdPlayer_PlayerInsertThing(t_py *x, int onset, PyObject *value) {
    if (x->playerDict == NULL) {
        x->playerDict = Py4pdPlayer_CreateDictionary();
    }
    int index = -1;
    for (int i = 0; i < x->playerDict->size; i++) {
        if (x->playerDict->entries[i].onset == onset) {
            index = i;
            break;
        }
    }

    if (index != -1) {
        // Onset already exists in the dictionary, add value to the array
        x->playerDict->entries[index].values = (PyObject **)realloc(
            x->playerDict->entries[index].values,
            (x->playerDict->entries[index].size + 1) * sizeof(PyObject *));
        x->playerDict->entries[index]
            .values[x->playerDict->entries[index].size] = value;
        x->playerDict->entries[index].size++;
    } else {
        // Onset doesn't exist, create a new entry
        if (onset > x->playerDict->lastOnset) {
            x->playerDict->lastOnset = onset;
        }
        x->playerDict->entries = (KeyValuePair *)realloc(
            x->playerDict->entries,
            (x->playerDict->size + 1) * sizeof(KeyValuePair));
        KeyValuePair *newEntry = &(x->playerDict->entries[x->playerDict->size]);
        newEntry->onset = onset;
        newEntry->values = (PyObject **)malloc(sizeof(PyObject *));
        newEntry->values[0] = value;
        newEntry->size = 1;
        x->playerDict->size++;
    }
}

// ======================================================
void Py4pdPlayer_FreeDictionary(Dictionary *dictionary) {
    if (dictionary == NULL) {
        return;
    }
    for (int i = 0; i < dictionary->size; i++) {
        free(dictionary->entries[i].values);
    }
    free(dictionary->entries);
    free(dictionary);
}

// ================== PLAYER ======================
void Py4pdPlayer_PlayTick(t_py *x) {
    if (x->playerDict->lastOnset > x->msOnset) {
        clock_delay(x->playerClock, 1);
    } else {
        clock_unset(x->playerClock);
    }
    x->msOnset++;
    KeyValuePair *entry = Py4pdPlayer_PlayerGetValue(x->playerDict, x->msOnset);
    if (entry != NULL) {
        for (int i = 0; i < entry->size; i++) {
            PyObject *pValue = Py_BuildValue("O", entry->values[i]);
            t_py4pd_pValue *pdPyValue =
                (t_py4pd_pValue *)malloc(sizeof(t_py4pd_pValue));
            pdPyValue->pValue = pValue;
            pdPyValue->objectsUsing = 0;
            pdPyValue->objOwner = x->objName;
            Py4pdUtils_ConvertToPd(x, pdPyValue, x->mainOut);
            free(pdPyValue);
        }
    }
}

// ======================================================
void Py4pdPlayer_Play(t_py *x, t_symbol *s, int ac, t_atom *av) {
    (void)s;
    x->msOnset = 0;
    if (x->playerDict == NULL) {
        pd_error(x, "[%s]: Nothing to play.", x->objName->s_name);
        return;
    }
    if (ac != 0 && av[0].a_type == A_FLOAT) {
        x->msOnset = (int)av->a_w.w_float;
    }
    x->playerClock = clock_new(x, (t_method)Py4pdPlayer_PlayTick);
    Py4pdPlayer_PlayTick(x);
    return;
}

// ======================================================
void Py4pdPlayer_Stop(t_py *x) {
    if (x->playerClock == NULL) {
        pd_error(x, "[%s]: Nothing to stop.", x->objName->s_name);
        return;
    }
    clock_unset(x->playerClock);
    return;
}

// ======================================================
void Py4pdPlayer_Clear(t_py *x) {
    if (x->playerClock != NULL) {
        clock_unset(x->playerClock);
    }
    Py4pdPlayer_FreeDictionary(x->playerDict);
    x->playerDict = NULL;
    return;
}
