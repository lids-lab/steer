use pyo3::prelude::*;

mod common;
mod fly;

#[pymodule]
fn _rustler(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<fly::Sampler>()?;

    Ok(())
}
