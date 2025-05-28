use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::HashMap;

/// Event struct representing connect or disconnect events
#[pyclass]
struct Event {
    #[pyo3(get, set)]
    event: String,
    #[pyo3(get, set)]
    trajectory: Option<usize>,
    #[pyo3(get, set)]
    t: Option<usize>,
}

#[pymethods]
impl Event {
    #[new]
    fn new(event: String, trajectory: Option<usize>, t: Option<usize>) -> Self {
        Event {
            event,
            trajectory,
            t,
        }
    }
}

/// Check if two points are within epsilon distance
fn check_epsilon_distance(p1: &[f64], p2: &[f64], eps: f64) -> bool {
    let distance = ((p1[0] - p2[0]).powi(2) + (p1[1] - p2[1]).powi(2) + (p1[2] - p2[2]).powi(2)).sqrt();
    distance <= eps
}

/// Find connect and disconnect events between two trajectories
#[pyfunction]
fn find_connect_disconnect_events<'py>(
    py: Python<'py>,
    t1_id: usize,
    t2_id: usize,
    t1: Vec<Vec<f64>>,
    t2: Vec<Vec<f64>>,
    eps: f64,
) -> PyResult<(&'py PyDict, &'py PyDict)> {
    let mut dic_t1: HashMap<usize, Vec<Event>> = HashMap::new();
    let mut dic_t2: HashMap<usize, Vec<Event>> = HashMap::new();
    
    let mut flag_t1 = vec![false; t1.len()];
    let mut flag_t2 = vec![false; t2.len()];
    
    let mut ti = 0;
    while ti < t1.len() {
        let mut tj = 0;
        while tj < t2.len() {
            if !flag_t1[ti] && !flag_t2[tj] && check_epsilon_distance(&t1[ti], &t2[tj], eps) {
                flag_t1[ti] = true;
                flag_t2[tj] = true;
                let mut flag_insert = true;
                let mut first_i = ti;
                let mut last_i = ti;
                let mut first_j = tj;
                let mut last_j = tj;
                
                while flag_insert && last_j < t2.len() {
                    flag_insert = false;
                    
                    // Case first_i - 1 insert
                    if first_i > 0 && !flag_t1[first_i - 1] {
                        let new_first_i = first_i - 1;
                        
                        if first_j > 0 && !flag_t2[first_j - 1] && check_epsilon_distance(&t1[new_first_i], &t2[first_j - 1], eps) {
                            first_i = new_first_i;
                            first_j -= 1;
                            flag_t1[first_i] = true;
                            flag_t2[first_j] = true;
                            flag_insert = true;
                        } else if last_j + 1 < t2.len() && !flag_t2[last_j + 1] && check_epsilon_distance(&t1[new_first_i], &t2[last_j + 1], eps) {
                            first_i = new_first_i;
                            last_j += 1;
                            flag_t1[first_i] = true;
                            flag_t2[last_j] = true;
                            flag_insert = true;
                        } else if first_j + 1 < t2.len() && !flag_t2[first_j + 1] && check_epsilon_distance(&t1[new_first_i], &t2[first_j + 1], eps) {
                            first_i = new_first_i;
                            first_j += 1;
                            flag_t1[first_i] = true;
                            flag_t2[first_j] = true;
                            flag_insert = true;
                        } else if last_j > 0 && !flag_t2[last_j - 1] && check_epsilon_distance(&t1[new_first_i], &t2[last_j - 1], eps) {
                            first_i = new_first_i;
                            last_j -= 1;
                            flag_t1[first_i] = true;
                            flag_t2[last_j] = true;
                            flag_insert = true;
                        } else {
                            for each_j in first_j..=last_j {
                                if check_epsilon_distance(&t1[new_first_i], &t2[each_j], eps) {
                                    first_i = new_first_i;
                                    flag_t1[first_i] = true;
                                    flag_t2[each_j] = true;
                                    flag_insert = true;
                                    break;
                                }
                            }
                        }
                    }
                    
                    // Case last_i + 1 insert
                    if (last_i + 1 < t1.len()) && !flag_t1[last_i + 1] {
                        let new_last_i = last_i + 1;
                        
                        if first_j > 0 && !flag_t2[first_j - 1] && check_epsilon_distance(&t1[new_last_i], &t2[first_j - 1], eps) {
                            last_i = new_last_i;
                            first_j -= 1;
                            flag_t1[last_i] = true;
                            flag_t2[first_j] = true;
                            flag_insert = true;
                        } else if last_j + 1 < t2.len() && !flag_t2[last_j + 1] && check_epsilon_distance(&t1[new_last_i], &t2[last_j + 1], eps) {
                            last_i = new_last_i;
                            last_j += 1;
                            flag_t1[last_i] = true;
                            flag_t2[last_j] = true;
                            flag_insert = true;
                        } else if first_j + 1 < t2.len() && !flag_t2[first_j + 1] && check_epsilon_distance(&t1[new_last_i], &t2[first_j + 1], eps) {
                            last_i = new_last_i;
                            first_j += 1;
                            flag_t1[last_i] = true;
                            flag_t2[first_j] = true;
                            flag_insert = true;
                        } else if last_j > 0 && !flag_t2[last_j - 1] && check_epsilon_distance(&t1[new_last_i], &t2[last_j - 1], eps) {
                            last_i = new_last_i;
                            last_j -= 1;
                            flag_t1[last_i] = true;
                            flag_t2[last_j] = true;
                            flag_insert = true;
                        } else {
                            for each_j in first_j..=last_j {
                                if check_epsilon_distance(&t1[new_last_i], &t2[each_j], eps) {
                                    last_i = new_last_i;
                                    flag_t1[last_i] = true;
                                    flag_t2[each_j] = true;
                                    flag_insert = true;
                                    break;
                                }
                            }
                        }
                    }
                    
                    // Case first_j - 1 insert
                    if first_j > 0 && !flag_t2[first_j - 1] {
                        let new_first_j = first_j - 1;
                        
                        if first_i > 0 && !flag_t1[first_i - 1] && check_epsilon_distance(&t2[new_first_j], &t1[first_i - 1], eps) {
                            first_j = new_first_j;
                            first_i -= 1;
                            flag_t1[first_i] = true;
                            flag_t2[first_j] = true;
                            flag_insert = true;
                        } else if last_i + 1 < t1.len() && !flag_t1[last_i + 1] && check_epsilon_distance(&t2[new_first_j], &t1[last_i + 1], eps) {
                            first_j = new_first_j;
                            last_i += 1;
                            flag_t1[last_i] = true;
                            flag_t2[first_j] = true;
                            flag_insert = true;
                        } else if first_i + 1 < t1.len() && !flag_t1[first_i + 1] && check_epsilon_distance(&t2[new_first_j], &t1[first_i + 1], eps) {
                            first_j = new_first_j;
                            first_i += 1;
                            flag_t2[first_j] = true;
                            flag_t1[first_i] = true;
                            flag_insert = true;
                        } else if last_i > 0 && !flag_t1[last_i - 1] && check_epsilon_distance(&t2[new_first_j], &t1[last_i - 1], eps) {
                            first_j = new_first_j;
                            last_i -= 1;
                            flag_t1[last_i] = true;
                            flag_t2[first_j] = true;
                            flag_insert = true;
                        } else {
                            for each_i in first_i..=last_i {
                                if check_epsilon_distance(&t2[new_first_j], &t1[each_i], eps) {
                                    first_j = new_first_j;
                                    flag_t2[first_j] = true;
                                    flag_t1[each_i] = true;
                                    flag_insert = true;
                                    break;
                                }
                            }
                        }
                    }
                    
                    // Case last_j + 1 insert
                    if (last_j + 1 < t2.len()) && !flag_t2[last_j + 1] {
                        let new_last_j = last_j + 1;
                        
                        if first_i > 0 && !flag_t1[first_i - 1] && check_epsilon_distance(&t2[new_last_j], &t1[first_i - 1], eps) {
                            last_j = new_last_j;
                            first_i -= 1;
                            flag_t1[first_i] = true;
                            flag_t2[last_j] = true;
                            flag_insert = true;
                        } else if last_i + 1 < t1.len() && !flag_t1[last_i + 1] && check_epsilon_distance(&t2[new_last_j], &t1[last_i + 1], eps) {
                            last_j = new_last_j;
                            last_i += 1;
                            flag_t1[last_i] = true;
                            flag_t2[last_j] = true;
                            flag_insert = true;
                        } else if first_i + 1 < t1.len() && !flag_t1[first_i + 1] && check_epsilon_distance(&t2[new_last_j], &t1[first_i + 1], eps) {
                            last_j = new_last_j;
                            first_i += 1;
                            flag_t1[first_i] = true;
                            flag_t2[last_j] = true;
                            flag_insert = true;
                        } else if last_i > 0 && !flag_t1[last_i - 1] && check_epsilon_distance(&t2[new_last_j], &t1[last_i - 1], eps) {
                            last_j = new_last_j;
                            last_i -= 1;
                            flag_t1[last_i] = true;
                            flag_t2[last_j] = true;
                            flag_insert = true;
                        } else {
                            for each_i in first_i..=last_i {
                                if check_epsilon_distance(&t2[new_last_j], &t1[each_i], eps) {
                                    last_j = new_last_j;
                                    flag_t2[last_j] = true;
                                    flag_t1[each_i] = true;
                                    flag_insert = true;
                                    break;
                                }
                            }
                        }
                    }
                }
                
                // Connect event
                dic_t1.entry(first_i).or_insert_with(Vec::new).push(Event {
                    event: "connect".to_string(),
                    trajectory: Some(t2_id),
                    t: Some(first_j),
                });
                
                dic_t2.entry(first_j).or_insert_with(Vec::new).push(Event {
                    event: "connect".to_string(),
                    trajectory: Some(t1_id),
                    t: Some(first_i),
                });
                
                // Disconnect event
                if last_i + 1 < t1.len() {
                    dic_t1.entry(last_i + 1).or_insert_with(Vec::new).push(Event {
                        event: "disconnect".to_string(),
                        trajectory: Some(t2_id),
                        t: Some(last_j + 1),
                    });
                }
                
                if last_j + 1 < t2.len() {
                    dic_t2.entry(last_j + 1).or_insert_with(Vec::new).push(Event {
                        event: "disconnect".to_string(),
                        trajectory: Some(t1_id),
                        t: Some(last_i + 1),
                    });
                }
                
                ti = last_i;
                break;
            } else {
                tj += 1;
            }
        }
        ti += 1;
    }
    
    // Convert Rust maps to Python dictionaries
    let py_dic_t1 = PyDict::new(py);
    for (key, events) in dic_t1 {
        let py_events = PyList::empty(py);
        for event in events {
            py_events.append(Py::new(py, event)?)?;
        }
        py_dic_t1.set_item(key, py_events)?;
    }
    
    let py_dic_t2 = PyDict::new(py);
    for (key, events) in dic_t2 {
        let py_events = PyList::empty(py);
        for event in events {
            py_events.append(Py::new(py, event)?)?;
        }
        py_dic_t2.set_item(key, py_events)?;
    }
    
    Ok((py_dic_t1, py_dic_t2))
}

/// A Python module implemented in Rust.
#[pymodule]
fn rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Event>()?;
    m.add_function(wrap_pyfunction!(find_connect_disconnect_events, m)?)?;
    Ok(())
} 