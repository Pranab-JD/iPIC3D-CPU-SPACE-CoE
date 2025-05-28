#include <iostream>
#include <filesystem>
#include <string>
#include <stdexcept>

namespace fs = std::filesystem;

void remove_all_contents(const fs::path& dir) {
    for (const auto& entry : fs::directory_iterator(dir)) {
        fs::remove_all(entry);
    }
}


/**
 * @brief prepare the output folder, create if needed
 * 
 * @details Check if the output folder exists,
 *          if not, create it. Make sure it's there in the end.
 *          Fail with exception.
 * 
 * @param relativePath: Output folder path relative to the current directory 
 * @returns 0 if succeed
 */

int checkOutputFolder(std::string relativePath)
{
    fs::path current_path = fs::current_path();
    fs::path subdir = current_path / relativePath;

    if (fs::exists(subdir)) 
    {
        if (fs::is_directory(subdir))
        {
            if (!fs::is_empty(subdir))
                remove_all_contents(subdir);
        }
        else throw std::runtime_error("Output path is not a directory");
    } 
    else
        fs::create_directories(subdir);

    return 0;

}