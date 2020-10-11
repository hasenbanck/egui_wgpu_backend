use shaderc::ShaderKind;
use std::error::Error;
use std::fs::{create_dir_all, metadata, read_dir, read_to_string, File};
use std::io::Write;
use std::path::Path;

const GLSL_PATH: &str = "src/shader";
const SPIRV_PATH: &str = "gen/shader";

fn main() -> Result<(), Box<dyn Error>> {
    println!("cargo:rerun-if-changed={}", GLSL_PATH);
    create_dir_all(SPIRV_PATH)?;

    let mut compiler = shaderc::Compiler::new().ok_or("can't create shaderc compiler")?;
    let mut options = shaderc::CompileOptions::new().unwrap();
    options.set_source_language(shaderc::SourceLanguage::GLSL);
    options.set_optimization_level(shaderc::OptimizationLevel::Performance);
    options.set_warnings_as_errors();

    for entry in read_dir(GLSL_PATH)? {
        let entry = entry?;
        let path = entry.path();
        let metadata = metadata(&path)?;

        if metadata.is_file() {
            let filename = path.file_name().ok_or("no filename")?.to_owned();
            let extension = path.extension().ok_or("no file extension")?.to_owned();

            if let Some(kind) = match extension.to_string_lossy().as_ref() {
                "vert" => Some(ShaderKind::Vertex),
                "frag" => Some(ShaderKind::Fragment),
                "comp" => Some(ShaderKind::Compute),
                _ => None,
            } {
                let source = read_to_string(&path)?;
                let spirv = compiler.compile_into_spirv(
                    &source,
                    kind,
                    filename.to_string_lossy().as_ref(),
                    "main",
                    Some(&options),
                )?;

                let spirv_path = Path::new(SPIRV_PATH)
                    .join(format!("{}.spv", filename.to_string_lossy().as_ref()));
                let mut spirv_file = File::create(spirv_path)?;
                spirv_file.write_all(spirv.as_binary_u8())?;
            }
        }
    }

    Ok(())
}
