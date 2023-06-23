#include "m_pd.h"
#include "py4pd.h"
#include "utils.h"  

// ======================================================
Dictionary* PY4PD_createDictionary() {
    Dictionary* dictionary = (Dictionary*)malloc(sizeof(Dictionary));
    dictionary->entries = NULL;
    dictionary->size = 0;
    dictionary->lastOnset = 0;
    return dictionary;
}

// ======================================================
void PY4PD_Player_InsertThing(t_py *x, int onset, PyObject *value) {
    if (x->playerDict == NULL) {
        x->playerDict = PY4PD_createDictionary();
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
        x->playerDict->entries[index].values = (PyObject**)realloc(x->playerDict->entries[index].values, (x->playerDict->entries[index].size + 1) * sizeof(PyObject*));
        x->playerDict->entries[index].values[x->playerDict->entries[index].size] = value;
        x->playerDict->entries[index].size++;
    } else {
        // Onset doesn't exist, create a new entry
        if (onset > x->playerDict->lastOnset){
            x->playerDict->lastOnset = onset;
        }
        x->playerDict->entries = (KeyValuePair*)realloc(x->playerDict->entries, (x->playerDict->size + 1) * sizeof(KeyValuePair));
        KeyValuePair* newEntry = &(x->playerDict->entries[x->playerDict->size]);
        newEntry->onset = onset;
        newEntry->values = (PyObject**)malloc(sizeof(PyObject*));
        newEntry->values[0] = value;
        newEntry->size = 1;
        x->playerDict->size++;
    }
}

// ======================================================
static int compareOnset(const void* a, const void* b) {
    KeyValuePair* kvp1 = (KeyValuePair*)a;
    KeyValuePair* kvp2 = (KeyValuePair*)b;
    
    if (kvp1->onset < kvp2->onset) {
        return -1;
    } else if (kvp1->onset > kvp2->onset) {
        return 1;
    } else {
        return 0;
    }
}

// ======================================================
KeyValuePair* PY4PD_Player_GetValue(Dictionary* dictionary, int onset) {
    if (dictionary == NULL) {
        return NULL;
    }
    KeyValuePair key;
    key.onset = onset;
    KeyValuePair* result = (KeyValuePair*)bsearch(&key, dictionary->entries, dictionary->size, sizeof(KeyValuePair), compareOnset);
    return result;  // Return NULL if no matching onset is found
}

// ======================================================
void PY4PD_freeDictionary(Dictionary* dictionary) {
    if (dictionary == NULL) {
        return;
    }
    for (int i = 0; i < dictionary->size; i++) {
        free(dictionary->entries[i].values);
    }
    free(dictionary->entries);
    free(dictionary);
}

// ================================================
// ================== PLAYER ======================
// ================================================

void py4pdPlay_tick(t_py *x){
    x->msOnset++; 
    if (x->playerDict->lastOnset > x->msOnset){
        clock_delay(x->playerClock, 1);
    }
    KeyValuePair* entry = PY4PD_Player_GetValue(x->playerDict, x->msOnset);
    if (entry != NULL) {
        for (int i = 0; i < entry->size; i++) {
            PyObject* value = entry->values[i];
            py4pd_convert_to_pd(x, value);
        }
    }
}

// ====================================================== 
void py4pdPlay(t_py *x){
    x->msOnset = 0;
    if (x->playerDict == NULL) {
        pd_error(x, "[%s]: Nothing to play.", x->objectName->s_name);
        return;
    }
    x->playerClock = clock_new(x, (t_method)py4pdPlay_tick);
    py4pdPlay_tick(x);

    return;
}


// ======================================================
void py4pdStop(t_py *x, t_symbol *s, int ac, t_atom *av){
    (void)s;
    (void)ac;
    (void)av;
    (void)x;
    clock_unset(x->playerClock);
    return;
}

// ======================================================
void py4pdClear(t_py *x){
    PY4PD_freeDictionary(x->playerDict);
    x->playerDict = NULL;
    return;
}
