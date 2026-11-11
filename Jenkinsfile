// Pipeline de integración continua del proyecto (Jenkins en Docker, ver docker-compose.yml).
//
// Etapas: preparar Python -> validar estructura -> entorno virtual -> entrenamiento ->
//         pruebas con cobertura -> monitoreo de drift -> SonarCloud -> imagen Docker de la API.
//
// Configuración del job: New Item > Pipeline > Definition: "Pipeline script from SCM" > Git >
// URL del repositorio, rama (*/developer, */certification o */master) y Script Path: Jenkinsfile.
//
// SonarCloud (opcional): crear en Manage Jenkins > Credentials una credencial "Secret text" con
// ID "sonarcloud-token" y el token generado en sonarcloud.io. Si no existe, la etapa se omite y
// el build queda como UNSTABLE (amarillo), sin fallar.

def dockerDisponible() {
    return sh(script: 'command -v docker > /dev/null 2>&1 && docker info > /dev/null 2>&1', returnStatus: true) == 0
}

pipeline {
    agent any

    options {
        timeout(time: 40, unit: 'MINUTES')
        buildDiscarder(logRotator(numToKeepStr: '10'))
    }

    stages {
        stage('Preparar Python') {
            // La imagen jenkins/jenkins no trae venv; el contenedor corre como root (docker-compose),
            // así que puede instalarlo la primera vez.
            steps {
                sh '''
                    if ! python3 -m venv --help > /dev/null 2>&1; then
                        apt-get update
                        apt-get install -y --no-install-recommends python3 python3-venv python3-pip
                    fi
                    python3 --version
                '''
            }
        }

        stage('Validar estructura') {
            steps {
                script {
                    // Archivos que exige la estructura del proyecto. Si falta alguno, el pipeline se detiene.
                    def estructura = [
                        'mlops_pipeline/Cargar_datos.ipynb',
                        'mlops_pipeline/comprension_eda.ipynb',
                        'mlops_pipeline/ft_engineering.py',
                        'mlops_pipeline/model_training_evaluation.py',
                        'mlops_pipeline/model_deploy.py',
                        'mlops_pipeline/model_monitoring.py',
                        'Base_de_datos.csv',
                        'requirements.txt',
                        '.gitignore',
                        'setup.bat',
                        'readme.md',
                    ]
                    def faltantes = estructura.findAll { !fileExists(it) }
                    if (faltantes) {
                        error("La estructura del repositorio no cumple. Faltan: ${faltantes.join(', ')}")
                    }
                    echo 'Estructura del repositorio correcta.'
                }
            }
        }

        stage('Entorno virtual') {
            steps {
                sh '''
                    python3 -m venv venv
                    venv/bin/pip install --upgrade pip
                    venv/bin/pip install -r requirements.txt
                '''
            }
        }

        stage('Entrenamiento') {
            // Genera mejor_modelo.joblib, que necesitan las pruebas de la API y el monitoreo.
            steps {
                sh 'venv/bin/python mlops_pipeline/model_training_evaluation.py'
            }
        }

        stage('Pruebas') {
            steps {
                sh 'venv/bin/python -m pytest mlops_pipeline -v --cov=mlops_pipeline --cov-report=xml --junitxml=resultados_pruebas.xml'
            }
            post {
                always { junit 'resultados_pruebas.xml' }
            }
        }

        stage('Monitoreo de drift') {
            steps {
                sh 'venv/bin/python mlops_pipeline/model_monitoring.py --job'
            }
        }

        stage('SonarCloud') {
            when { expression { return dockerDisponible() } }
            steps {
                script {
                    try {
                        withCredentials([string(credentialsId: 'sonarcloud-token', variable: 'SONAR_TOKEN')]) {
                            // El scanner corre en su propio contenedor y ve el workspace gracias a
                            // --volumes-from jenkins (mismo volumen y misma ruta que el contenedor de Jenkins).
                            sh '''
                                docker run --rm --volumes-from jenkins -u 0 \
                                    -e SONAR_TOKEN -e SONAR_USER_HOME=/tmp/sonar \
                                    -w "$WORKSPACE" sonarsource/sonar-scanner-cli \
                                    -Dsonar.projectBaseDir="$WORKSPACE"
                            '''
                        }
                    } catch (err) {
                        unstable("No se ejecutó el análisis de SonarCloud: ${err.getMessage()}")
                    }
                }
            }
        }

        stage('Imagen Docker') {
            when { expression { return dockerDisponible() } }
            steps {
                sh 'docker build -t riesgo-impago-api:${BUILD_NUMBER} -t riesgo-impago-api:latest .'
            }
        }
    }

    post {
        always {
            archiveArtifacts allowEmptyArchive: true, fingerprint: true, artifacts:
                'mlops_pipeline/mejor_modelo.joblib, mlops_pipeline/resumen_evaluacion.csv, ' +
                'mlops_pipeline/comparacion_modelos.png, mlops_pipeline/monitoreo_*.csv, coverage.xml'
        }
    }
}
